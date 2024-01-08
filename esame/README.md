# CAR PRICE PREDICTION


## INSTALL KIND ON UBUNTU
```bash 
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

## CREATE CLUSTER ON KUBERNETES
`kind create cluster --config=node_config.yaml`

### CHECK NODES
`kubectl get nodes`

### INSTALLING THE KUBERNETES DASHBOARD
`kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml`

### CONFIGURATION OF THE DASHBOARD
```bash 
kubectl apply -f dashboard_admin.yaml
kubectl apply -f cluster_role_binding.yaml
```
### GET THE TOKEN TO LOGIN AS ADMIN

`kubectl -n kubernetes-dashboard create token admin-user`

### ACCESS TO THE DASHBOARD

`kubectl proxy`

-Now you can insert the token create before to access to the kubernetes dashboard following:
[Accesso alla Dashboard di Kubernetes](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/)




## CREATE THE IMAGE ON DOCKER AND PUSHING ON DOCKER HUB
```bash
docker build --tag load_data_car load_data
docker tag load_data_car ossalag00/load_data_car
docker push docker.io/ossalag00/load_data_car 

docker build --tag create_features_car create_features
docker tag create_features_car ossalag00/create_features_car
docker push docker.io/ossalag00/create_features_car 

docker build --tag g_boost_model g_boost
docker tag g_boost_model ossalag00/g_boost_model
docker push docker.io/ossalag00/g_boost_model 

docker build --tag linear_regression_model linear_regression
docker tag linear_regression_model ossalag00/linear_regression_model
docker push docker.io/ossalag00/linear_regression_model

docker build --tag random_forest_model random_forest
docker tag random_forest_model ossalag00/random_forest_model
docker push docker.io/ossalag00/random_forest_model
```

## INSTALLATION OF KUBEFLOW PIPELINE 
```bash
export PIPELINE_VERSION=2.0.3
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```

-Verify that the Kubeflow Pipelines UI is accessible by port-forwarding:

`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`


### OPEN THE KUBEFLOW DASHBOARD
[kubeflow](http://localhost:8080/)

### CREATE THE PIPELINE ON KUBEFLOW
-Once you have access to the kubeflow dashboard, you can create and launch the pipeline by following the following steps:
    -In the Pipeline section click on create new pipeline;
    -Load the pipeline from the file (in our case **car_price_pipeline.yaml**) and create the pipeline
    -Create a new run of it and run it

## RUN THE APP LOCALLY
### REQUIREMENTS

`pip install requirements_app.txt`

-To run locally our app you need to launch:

`streamlit run app.py`

## DEPLOY ON KUBERNETES
-Before deploy we need to create the image of the app, and pushing on the hub with:

```bash
docker build --tag car_price_app .
docker tag car_price_app ossalag00/car_price_app
docker push docker.io/ossalag00/car_price_app
```
-To deploy:

`kubectl create --filename k8s_car_price_deployments.yaml`

-To see the deployments, pods or service:
```bash
kubectl get deployments
kubectl get pods
kubectl get services
```
### TO SEE THE APP ON KUBERNETES
[APP](http://localhost:30070)