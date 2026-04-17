
__all__ = [
    '_patch_priority',
    '_patch_MatrixLinearOperator_matmat',
]

def _patch_priority(classes, priority_name, replaced_operators):
    """Patch new prioritys..
    
    This is necessary to use ``__r*__`` methods like ``__rmatmul__`` from custom classes
    instead of ``__*__`` methods like ``__matmul__`` from the given classes if they never return :const:`NotImplemented`.
    
    This function replaces operators like ``__matmul__`` or ``__eq__`` of
    classes. In the new operator it is first 
    checked if ``other`` has the attribute ``priority_name`` with a higher value than 
    the classes's attribute ``priority_name`` which is set to 0.0. If so, :const:`NotImplemented`
    is returned. Otherwise the original implementation of the operator is called.

    This workaround is based on:
    https://github.com/scipy/scipy/issues/4819#issuecomment-920722279
    """
    
    def teach_priority(operator):
        def respect_priority(self, other):
            if getattr(self, priority_name) < getattr(other, priority_name, -1.0):
                return NotImplemented
            else:
                return operator(self, other)
        return respect_priority

    for cls in classes:
        if not hasattr(cls, priority_name):
            setattr(cls, priority_name, 0.0)
        
        for operator_name in replaced_operators:
            if hasattr(cls, operator_name):
                operator = getattr(cls, operator_name)
                check_if_applied = '__priority_patch_applied' + priority_name
                if not getattr(operator, check_if_applied, False):
                    wrapped_operator = teach_priority(operator)
                    setattr(wrapped_operator, check_if_applied, True)
                    setattr(cls, operator_name, wrapped_operator)


def _patch_MatrixLinearOperator_matmat():
    """ Patch MatrixLinearOperator ``@`` operation, as it does not work for multiplying
     MatrixLinearOperator based on a numpy array with a non-LinearOperator and non-numpy array (e.g. sparse array) right now.
    """
    from scipy.sparse.linalg._interface import MatrixLinearOperator
    import numpy as np
    
    _matmat =  getattr(MatrixLinearOperator, '_matmat')
    if not getattr(_matmat, '__matmat_patch_applied', False):
        def _matmat_new(self, X):
            return self.A @ X
        _matmat_new.__matmat_patch_applied = True
        setattr(MatrixLinearOperator, '_matmat', _matmat_new)