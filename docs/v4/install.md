# Install an alpha version of Parcels v4

````{warning}
Before installing an alpha version of Parcels, we *highly* recommend creating a new environment so that doesn't affect package versions in your current environment (which you may be using for your research).

Do the following to create a new environment:

```sh
conda create -n parcels-v4-alpha python=3.11
conda activate parcels-v4-alpha
```

````

During development of Parcels v4, we are uploading versions of the package to an [index on prefix.dev](https://prefix.dev/channels/parcels/packages/parcels). This allows users to easily install an unreleased version without having to do a [development install](../installation.rst)! Give it a spin!

```sh
conda install -c https://repo.prefix.dev/parcels parcels
```

During the development of Parcels v4 we will be occasionally releasing these alpha package versions so that users can try them out. If you're installing Parcels normally (i.e., via Conda forge) you can continue to do so without disruption.
