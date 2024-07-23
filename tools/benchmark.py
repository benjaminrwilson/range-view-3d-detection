"""Run benchmarking."""

import logging
import math
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Final, List, Tuple, cast

import matplotlib.pyplot as plt
import polars as pl
import torch
from av2.evaluation.detection.eval import DetectionCfg, evaluate
from draw_utils import draw_detections
from pytorch_lightning import LightningModule
from torchvision.io.image import write_png
from tqdm import tqdm

from torchbox3d.evaluation.evaluate import evaluate_waymo
from torchbox3d.math.ops.coding import build_dataframe
from torchbox3d.nn.arch.detector import Detector, prepare_for_evaluation
from torchbox3d.nn.decoders.range_decoder import RangeDecoder
from torchbox3d.prototype.loader import DataModule, _collate_fn
from torchbox3d.utils.wandb import load_artifact

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

UUID_KEYS: Final = ("log_id", "timestamp_ns")


def benchmark(
    model_name: str,
    project_name: str,
    entity: str,
    version: int,
    device_type: str = "cuda",
    dtype: torch.dtype = torch.float32,
    num_warmup: int = 5,
    log_interval: int = 50,
    network_only: bool = True,
    log_outputs: bool = True,
    use_oracle: bool = False,
) -> None:
    """Training entrypoint.

    Args:
        cfg: Training configuration.
    """
    # Set start method to forkserver to avoid fork related issues.
    torch.multiprocessing.set_forkserver_preload(["torch"])
    torch.multiprocessing.set_start_method("forkserver")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    if use_oracle:
        logger.warning("USING ORACLE FOR EVALUATION!")

    path = load_artifact(model_name, project_name, entity, version)

    model: LightningModule = Detector.load_from_checkpoint(path).eval()
    datamodule = DataModule.load_from_checkpoint(path)

    # dataloader = data.DataLoader(
    #     datamodule.val_dataset,
    #     batch_size=1,
    #     num_workers=1,
    #     collate_fn=_collate_fn,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )
    # dataloader = datamodule.val_dataloader()
    # dataloader.batch_size = 1

    backbone = model.backbone
    head = model.head
    decoder = cast(RangeDecoder, model.decoder)

    dataset_name = cast(str, model.dataset_name)

    subsampling_rate = 1000
    indices = range(len(datamodule.val_dataset))[::subsampling_rate]

    metrics_list: List[pl.DataFrame] = []
    min_confidences = (0.1,)
    breakdowns = defaultdict(list)
    logger.info("Starting trial model name: %s.\n", model_name)
    for t in min_confidences:
        # logger.info("Starting trial with minimum confidence: %s.\n", t)
        pure_inf_time = 0.0
        decoder.enable_sample_by_range = True
        # model.post_processing_config["nms_mode"] = "WEIGHTED"
        model.post_processing_config["nms_threshold"] = 0.3
        model.post_processing_config["min_confidence"] = 0.1
        # model.post_processing_config["num_pre_nms"] = 4096
        # model.post_processing_config["num_post_nms"] = 1000
        logger.info("Using post processing config: %s.\n", model.post_processing_config)
        logger.info("Using decoder: %s.\n", decoder)

        dts_list = []
        uuids_list = []
        for i, index in enumerate(tqdm(indices)):
            # if i % 100 != 0:
            #     continue
            datum = datamodule.val_dataset[index]
            datum = {
                k: v.to(device_type) if isinstance(v, torch.Tensor) else v
                for k, v in _collate_fn([datum]).items()
            }

            uuids = cast(pl.DataFrame, datum["uuids"]).with_columns(batch_index=0)
            with torch.autocast(device_type=device_type, dtype=dtype):
                backbone_outputs, neck_elapsed = bench(backbone, datum)
                (head_outputs, loss), head_elapsed = bench(
                    head,
                    backbone_outputs,
                    datum,
                    return_loss=use_oracle,
                )
                outputs = {
                    "neck": backbone_outputs,
                    "head": head_outputs,
                }
                if "aux" in loss:
                    outputs["aux"] = loss["aux"]

                (params, scores, categories, batch_index), decoder_elapsed = bench(
                    decoder.decode,
                    head_outputs,
                    post_processing_config=model.post_processing_config,
                    task_config=model.tasks,
                    use_nms=True,
                    data=datum,
                )

            elapsed = neck_elapsed + head_elapsed + decoder_elapsed
            breakdowns["t"] += [t]
            breakdowns["backbone"] += [neck_elapsed * 1e3]
            breakdowns["head"] += [head_elapsed * 1e3]
            breakdowns["decoder"] += [decoder_elapsed * 1e3]
            breakdowns["total"] += [elapsed * 1e3]

            if i >= num_warmup:
                pure_inf_time += elapsed
                fps = (i + 1 - num_warmup) / pure_inf_time

            dts = build_dataframe(
                params,
                scores,
                categories,
                batch_index,
                uuids,
                model.category_id_to_category,
            )

            gts = cast(pl.DataFrame, datum["annotations"])
            if log_outputs:
                output_dir = Path.home() / "data" / "predictions" / dataset_name
                output_dir.mkdir(exist_ok=True, parents=True)

                img = draw_detections(
                    dts=dts,
                    uuids=uuids,
                    data=datum,
                    network_outputs=outputs,
                    max_side=50.0,
                )
                log_id, timestamp_ns, _ = uuids.filter(pl.col("batch_index") == 0).row(
                    0
                )

                dst = output_dir / f"{log_id}-{timestamp_ns}.png"
                write_png(img, str(dst))

                # breakpoint()

            dts_list.append(dts)
            uuids_list.append(uuids)

        dts = pl.concat(dts_list)

        breakdown_averages = pl.DataFrame(breakdowns)[num_warmup:].mean()
        dataset_dir = datamodule.val_dataset.split_dir

        valid_categories = list(chain.from_iterable(model.tasks.values()))
        metadata = pl.concat(uuids_list).drop("batch_index")
        dts, gts, cfg = prepare_for_evaluation(
            dts,
            model.dataset_name,
            split=model.evaluation_split_name,
            dataset_dir=dataset_dir,
            valid_categories=valid_categories,
            metadata=metadata,
        )
        dts, gts, metrics = evaluate_dispatch(
            dts,
            gts,
            dataset_name=dataset_name,
            cfg=cfg,
        )

        metrics = metrics.with_columns(
            t=t,
            backbone=breakdown_averages["backbone"],
            head=breakdown_averages["head"],
            decoder=breakdown_averages["decoder"],
            total=breakdown_averages["total"],
            fps=fps,
        )
        metrics_list.append(metrics)

    if dataset_name == "av2":
        average_metrics = metrics.filter(pl.col("category") == "AVERAGE_METRICS")
        average_metrics = average_metrics.with_columns(
            AP=100 * pl.col("AP"), CDS=100 * pl.col("CDS")
        )
    elif dataset_name == "waymo":
        metrics = (
            metrics.filter((pl.col("r_lower") == 0.0) & (pl.col("r_upper") == math.inf))
            .filter((pl.col("type") == "3D") & (pl.col("level") == 1))
            .filter(pl.col("category") != "SIGN")
            .sort("category")
        )
        average_metrics = pl.concat(
            (
                pl.DataFrame({"category": "AVERAGE_METRICS"}),
                metrics.select(
                    pl.col("value").cast(pl.Float64).round(3).alias("AP")
                ).mean(),
                metrics.select(pl.col("fps").round(3)).mean(),
            ),
            how="horizontal",
        )
    logger.info("FULL RESULTS: %s", metrics)
    logger.info("AVERAGE RESULTS: %s", average_metrics)

    y = average_metrics["AP"].to_numpy()
    x = average_metrics["fps"].to_numpy()

    plt.plot(x, y)
    plt.title("Runtime vs. Performance")
    plt.xlabel("Hz")
    plt.ylabel("Average Precision")
    plt.savefig(
        f"performance_vs_runtime-{model_name}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1200,
    )


def evaluate_dispatch(
    dts: pl.DataFrame,
    gts: pl.DataFrame,
    dataset_name: str,
    cfg: DetectionCfg,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Dispatch to dataset-specific evaluation benchmark."""
    if dataset_name.upper() == "AV2":
        dts_pandas, gts_pandas, metrics = evaluate(
            dts=dts.to_pandas(), gts=gts.to_pandas(), cfg=cfg
        )

        dts = pl.from_pandas(dts_pandas)
        gts = pl.from_pandas(gts_pandas)
        metrics = pl.from_pandas(metrics.reset_index(names="category"))
    elif dataset_name.upper() == "WAYMO":
        metrics = evaluate_waymo(dts, gts)
    else:
        raise NotImplementedError("This dataset is not implemented.")
    return dts, gts, metrics


@torch.no_grad()
def bench(fn: Callable, *datum, **kwargs) -> Tuple[Any, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = fn(*datum, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output, elapsed


if __name__ == "__main__":
    benchmark(
        model_name="oibdid6c",
        project_name="av2-release",
        entity="benjaminrwilson",
        version=0,
        dtype=torch.float16,
        use_oracle=False,
        log_outputs=False,
    )

# bn7lkwep - Waymo Large
