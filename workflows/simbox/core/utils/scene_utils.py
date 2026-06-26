import random


def deactivate_selected_prims(prim, selected_names, random_names):
    for child_prim in prim.GetAllChildren():
        for name in selected_names:
            if name in child_prim.GetName().lower():
                child_prim.SetActive(False)
                print(f"Deactivating: {child_prim.GetPath()}")

        for name in random_names:
            if name in child_prim.GetName().lower():
                flag = random.random() > 0.5
                child_prim.SetActive(flag)
                if not flag:
                    print(f"Deactivating: {child_prim.GetPath()}")

        deactivate_selected_prims(child_prim, selected_names, random_names)
