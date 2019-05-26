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

def check_mv_method(method, method_options, control, variant, variants, lift=0):
    if not method in method_options:
        raise ValueError("Method '{}' is not a valid method. "
                         "Available methods are {}."
                         .format(method, method_options))

    if not control is None:
        if not control in variants:
            raise ValueError("Control variant '{}' not available. "
                "Variants = {}.".format(control, variants))

    if not variant in variants:
        raise ValueError("Variant '{}' not available. "
                "Variants = {}.".format(variant, variants))

    if not isinstance(lift, numbers.Number) or lift < 0:
        raise ValueError("Lift must be a positive number;"
            " got lift={}".format(lift))

    if lift > 0 and method != "MC":
        raise ValueError("Method {} cannot be used with lift={}."
            " Select method='MC'.".format(method, lift))


def check_models(refclass, *models):
    """
    Check that models for A/B and multivariate testing belong to the correct
    class.

    Parameters
    ----------
    refclass : object
        Reference class.

    models : objects
        Model instances to be checked.
    """
    for model_id, model in enumerate(models):
        if not isinstance(model, refclass):
            raise TypeError("Model {} is not an instance of {}."
                .format(model_id, refclass.__name__))


def check_mv_models(refclass, models):
    """
    Check models for Multivariate testing.

    Parameters
    ----------
    refclass : object
        Reference class.

    models : dict
        Dictionary of model instances to be checked.
    """
    if not isinstance(models, dict):
        raise TypeError("Input models must be of type dict.")

    variants = models.keys()
    variant_control = "A"

    if variant_control not in variants:
        raise ValueError("A model variant 'A' (control) is required.")

    model_classes = models.values()
    check_models(refclass, *model_classes)
