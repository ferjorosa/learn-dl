### When i test my model, do I have to use `model.eval()` even though I am using `with torch.no_grad()` ?

<a href="https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/38">Question link</a>

These two have different goals:
* `model.eval()` will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
* `torch.no_grad()` impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).

This means that during evaluation, it is advisable to do:

```
model.eval()
with torch.no_grad():
    for batch in val_loader:
        #some code
```

* `model.eval()` is enough to get valid results.
* `torch.no_grad()` will additionally save some memory.

