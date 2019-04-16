import numbers


def check_ab_method(method, method_options, variant, lift=0):
    """
    Check parameters of A/B method.

    Parameters
    ----------
    method : str
        The default computational method.

    method_options : list or tuple
        The list of supported computational methods.

    variant : str
        The chosen variant. Options are "A", "B", "all"

    lift : float (default=0.0)
        The amount of uplift.
    """
    if not method in method_options:
        raise ValueError("Method '{}' is not a valid method. "
                         "Available methods are {}."
                         .format(method, method_options))

    if not variant in ("A", "B", "all"):
        raise ValueError("Variant must be 'A', 'B' or 'all'.")
    
    if not isinstance(lift, numbers.Number) or lift < 0:
        raise ValueError("Lift must be a positive number;"
            " got lift={}".format(lift))

    if lift > 0 and method != "MC":
        raise ValueError("Method {} cannot be used with lift={}."
            " Select method='MC'.".format(method, lift))


def check_models(classmodel, *models):
    """
    Check models class.

    Parameters
    ----------
    classmodel : object

    models : 
    """
    for model_id, model in enumerate(models):
        if not isinstance(model, classmodel):
            raise TypeError("Model {} is not an instance of {}."
                .format(model_id, classmodel.__class__.__name__))