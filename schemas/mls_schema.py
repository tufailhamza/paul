from pydantic import BaseModel, validator
from typing import Optional
import re

class MLSRecord(BaseModel):
    MLS: int
    PicCount: Optional[int]
    Status: str
    Type: str
    L_Price: str
    Address: str
    District: Optional[str]
    Bds: Optional[int]
    Bths: Optional[float]
    Liv_SF: Optional[int]
    City: str
    State: str
    Zip: str

    @validator("Status")
    def status_must_be_valid(cls, v):
        valid_status = {"ACT", "SLD", "EXP", "CXL", "PEND"}
        if v.upper() not in valid_status:
            raise ValueError(f"Invalid Status: {v}")
        return v

    @validator("L_Price")
    def price_must_be_number(cls, v):
        # Remove $ and commas, ensure numeric
        clean = re.sub(r"[^\d.]", "", v)
        if not clean.isdigit():
            raise ValueError(f"Invalid Price: {v}")
        return int(clean)

    @validator("Bds", "Bths", "Liv_SF", pre=True, always=True)
    def optional_numeric(cls, v):
        if v is None or v == "":
            return None
        return float(v)
