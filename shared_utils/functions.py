import math
import numpy as np
import datetime
import decimal
import logging

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

def clean_sample_data(data: list[dict]) -> list[dict]:
    def convert(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        elif isinstance(v, (np.integer, np.floating)):
            return v.item()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (datetime.datetime, datetime.date)):
            return v.isoformat()
        elif isinstance(v, decimal.Decimal):
            return float(v)
        return v

    return [{k: convert(v) for k, v in row.items()} for row in data]

def make_json_safe(data):
    """Clean any NaN/inf/-inf issues from sample_data before JSON use"""
    def clean_value(v):
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
        elif isinstance(v, str):
            if v.lower() in {"nan", "inf", "-inf"}:
                return None
        return v

    return [
        {k: clean_value(v) for k, v in row.items()}
        for row in data
    ]


