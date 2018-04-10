import imp


def set_missing_cf_field(field_name, cf, default_value):
    if not hasattr(cf, field_name):
        setattr(cf, field_name, default_value)


def load_config(config_path, name='config'):
    cf = imp.load_source(name, config_path)
    return cf
