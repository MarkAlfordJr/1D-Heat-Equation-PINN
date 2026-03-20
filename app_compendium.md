# App Compendium

for classes/modularity, each file should have a class
this class has:
    - States (properties): that should stay together
    - Behaviors (methods): that act upon those states

when calling the class, you make an object
object = class(state1, state2, etc)
object.behavior1

## Modularity Guidelines

file.py is the module
- the module has a class that does ONE job only

- for Methods that must use other files data, use the module file as an arg, then we you eventually call the methods in run_experiment.py, use the objects for those module classes.

## direct coupling dependency
- model.py should usually import almost nothing from your own project
- physics.py may depend on model
- loss.py may depend on physics
- trainer.py may depend on loss
- run_experiment.py wires them all together