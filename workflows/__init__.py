# flake8: noqa: F401
# pylint: disable=W0611
def import_extensions(workflow_type):
    if workflow_type == "SimBoxDualWorkFlow":
        import workflows.simbox_dual_workflow
    else:
        raise ValueError(f"Unsupported workflow type: {workflow_type}")
