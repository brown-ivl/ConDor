# Training the model 



### Loading the environment

1. ```
   conda env create -f environment.yaml
   ```

2. ```
   conda activate ConDor
   ```



### Partiality

In ```cfgs/config_capsules_multi.yaml``` change the parameter ```feature.partiality.use``` to `true`.

```
python3 main.py
```



### Full shapes

In ```cfgs/config_capsules_multi.yaml``` change the parameter ```feature.partiality.use``` to `false`.

```
python3 main.py
```



