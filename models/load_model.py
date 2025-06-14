from models.stock_attention import StockAttentioner

def load_model(args):
    if args.model_name == 'StockAttentioner':
        return StockAttentioner(
            args
        )

if __name__ == "__main__":
    import yaml
    from types import SimpleNamespace

    def load_config(path):
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return SimpleNamespace(**cfg_dict)

    args = load_config("config/config.yaml")
    args.num_stocks = 49
    
    print(args)
    load_model(args)
        
    