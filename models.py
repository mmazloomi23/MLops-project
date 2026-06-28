from pydantic import BaseModel, Field
from typing import List

class OrderDeliveryInput(BaseModel):
    order_id: str = Field(..., description="Unique identifier for the order")
    purchase_hour: int
    purchase_dayofweek: int
    purchase_month: int
    is_weekend_purchase: int
    estimated_delivery_days: float
    has_approval_timestamp: int
    approval_delay_hours: float
    num_items: int
    total_price: float
    avg_price: float
    max_price: float
    total_freight: float
    avg_freight: float
    max_freight: float
    num_sellers: int
    num_products: int
    num_product_categories: int
    avg_product_weight_g: float
    max_product_weight_g: float
    avg_product_volume_cm3: float
    max_product_volume_cm3: float
    main_product_category: str
    freight_price_ratio: float
    payment_value: float
    payment_installments: int
    num_payment_types: int
    main_payment_type: str
    seller_zip_code_prefix: float
    num_seller_states: int
    main_seller_city: str
    main_seller_state: str
    customer_zip_code_prefix: float
    customer_city: str
    customer_state: str
    same_state: int
    same_city: int
    zip_prefix_diff: float

class PredictionOutput(BaseModel):
    order_id: str
    late_probability: float
    predicted_is_late: bool
    risk_level: str
    model_version: str
    recommended_action: str