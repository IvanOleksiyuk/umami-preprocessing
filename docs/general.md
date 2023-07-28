Here we will give the explanation to the options given in the config file 


first we advise to specify the location of the TDD files and the names of the companent as
```

```

##global cuts

##outlier cuts



| Setting | Type | Explanation |
| ------- | ---- | ----------- |
| `fraction` | `float` | 1 will yield as many jets after sampling as there were in the dataset (because of ovesampling) other values wil yield proportionally smaller or larger fraction |


It is important to provide the variables one would like to have in the training files.
this can be done either by adding them manually in the body of the config like this
```
variables:
  jets:
    inputs:
      - var1
      - var2
    labels:
      - l1
      - l2
  tracks:
    inputs:
      - var3
      - var4
    labels:
      - l3
      - l4
```
Or by writing the in a separate config file **in the same directory** nd include it in the main config file using:

```
variables: !include <your variables file>.yaml
```