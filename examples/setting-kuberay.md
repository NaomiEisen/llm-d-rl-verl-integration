All the steps above are taken from here:
[KubeRay Operator Helm Chart](https://github.com/ray-project/kuberay/tree/7092f76e6f08fa86ad21c37cd8216914dd215975/helm-chart/kuberay-operator)

I chose the approach of installing the CRDs separately from the operator, to keep the permission requirements for each independent.

### 1. Add the KubeRay Helm repository

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
```
To see whether the cluster you're working on has them, run:

```bash
oc get crd | grep ray
```

You should see:

```
rayclusters.ray.io
rayjobs.ray.io
rayservices.ray.io
```

If not, the CRDs can be installed with:

```bash
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/crd?ref=v1.5.1&timeout=90s"
```

But this requires admin.

### 2. Install the operator specifically into your namespace

```bash
helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.5.1 \
  --namespace <your-namespace> \
  --set singleNamespaceInstall=true \
  --skip-crds
```

Note that I used the flags for deploying within the namespace, and I skipped the CRD installation. This will also enable non-admin users to deploy it, in clusters that already have the CRDs.
