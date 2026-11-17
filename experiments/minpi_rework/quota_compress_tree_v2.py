"""Corrected sparse-forest entry point for new studies. The signed 13-source replay uses v1."""
from . import quota_compress_tree as archived
from .quota_compress_tree_fidelity import gain_prefix


class Forest(archived.Forest):
    def gain_prefix(self,x,order,rho):
        cols=gain_prefix(self,x,order,rho)
        return cols,dict(checks=len(order)+1)


def extract(model,row):
    return Forest(**vars(archived.extract(model,row)))
