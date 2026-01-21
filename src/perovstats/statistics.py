import logging
from pathlib import Path
from yaml import safe_dump

import pandas as pd
from .classes import ImageData
from loguru import logger

def save_to_csv(config: dict[str, any], data: ImageData) -> None:
    rows = []

    parent_dict = data.__dict__
    grains_dict = parent_dict.pop("grains")

    for grain in grains_dict.values():
        row = {
            **grain.__dict__,
            **parent_dict
        }

        # Remove datatypes unfit for csv
        for key in ("mask", "config", "mask_rgb", "low_pass", "high_pass"):
            row.pop(key, None)

        rows.append(row)

    output_filename = f"output/{data.filename}/{data.filename}_statistics.csv"

    df = pd.DataFrame(rows)
    df.to_csv(output_filename, index=False)

    # Save configuration metadata for frequency splitting
    with Path(data.file_directory / f"{data.filename}_config.yaml").open("w") as outfile:
        safe_dump(config, outfile, default_flow_style=False)

    logger.info(
            f"exported statistics to {output_filename} along with its configuration settings.",
        )
