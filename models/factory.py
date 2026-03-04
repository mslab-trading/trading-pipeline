# models/factory.py
from models.stock_attentioner import StockAttentioner
from models.base import BaseModel

MODEL_REGISTRY = {
    "StockAttentioner": StockAttentioner,
}

def load_main_model(args, preprocessor_model=None):
    try:
        cls = MODEL_REGISTRY[args.model_name]
    except KeyError:
        raise ValueError(f"Unknown main model {args.model_name}")
    main_model = cls(args)
    complete_model = BaseModel(main_model, preprocessor_model)
    return complete_model
        
    