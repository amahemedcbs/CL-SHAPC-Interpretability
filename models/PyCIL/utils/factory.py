def get_model(model_name, args):
    name = model_name.lower()
    if name == "der":
        from ..models.der import DER
        return DER(args)
    elif name == "foster":
        from ..models.foster import FOSTER
        return FOSTER(args)
    elif name == "memo":
        from ..models.memo import MEMO
        return MEMO(args)
    elif name == "tagfex":
        from ..models.tagfex import TagFex
        return TagFex(args)
    else:
        assert 0
