import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.config import (
    default_float,
    set_default_float,
    set_default_summary_fmt,
)
from gpflow.utilities import ops, print_summary
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
set_default_float(np.float64)
import pandas as pd
import math
from gpflow import covariances, kernels, likelihoods
from gpflow.base import InputData, MeanAndVariance, OutputData, Parameter, RegressionData, TensorType
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.utilities import assert_params_false, positive, to_default_float
from gpflow.utilities.ops import pca_reduce
from gpflow.models.gpr import GPR
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import InducingVariablesLike, data_input_to_tensor, inducingpoint_wrapper

#数据预处理
file="D:/数据集(2)/li ML（200）.csv"
dataset=pd.read_csv(file)
print(dataset.shape[0])
print(dataset.shape[1])
dataset.head()
y=np.array(dataset["Target"])
x=np.array(dataset.iloc[:,2:65])
x=tf.convert_to_tensor(x, dtype=default_float())
y=tf.convert_to_tensor(y, dtype=default_float())
s_scale = StandardScaler()
s_scale.fit(x)
x= s_scale.transform(x)

#GPLVM
latent_dim = 2  # number of latent dimensions
num_inducing = 0  # number of inducing pts 设置为0   
num_data = x.shape[0]  # number of data points
#initialize via PCA
z_mean_init = ops.pca_reduce(x, latent_dim)
z_var_init = tf.ones((num_data, latent_dim), dtype=default_float())
#Pick inducing inputs randomly from dataset initialization
np.random.seed(1)  # for reproducibility
inducing_variable = tf.convert_to_tensor(
    np.random.permutation(z_mean_init.numpy())[:num_inducing],
    dtype=default_float(),
)
nnkernel=gpflow.kernels.ArcCosine(variance=1)
gplvm = gpflow.models.GPLVM(
    x,
    X_data_mean=z_mean_init,
    kernel=nnkernel,
    latent_dim=2,
)

gplvm.likelihood.variance.assign(0.01)
#optimize the created model,given that this model has a determinisitic evidence lower bound,use SciPy's BFGS to optimize
opt = gpflow.optimizers.Scipy()
maxiter = reduce_in_tests(100)
opt.minimize(
    gplvm.training_loss,
    method="BFGS",
    variables=gplvm.trainable_variables,
    options=dict(maxiter=100)
)

#生成虚拟样本
gplvm_z_mean=gplvm.data[0].numpy()#()
#gplvm_z_var=gplvm.X_data_var.numpy()
z=gplvm_z_mean#隐变量
z1=z.transpose()
inv=np.linalg.inv(np.dot(z1,z))
W=np.dot(inv,np.dot(z1,x)).transpose()
z_mean=np.mean(z,axis=0)
rng=check_random_state(0)
var=np.var(z,axis=0)
cov=var*np.eye(2)
samples_var=[]
for i in range(z.shape[0]):
    znew_samples_var=rng.multivariate_normal(z[i],cov,3)
    samples_var.append(znew_samples_var)
samples_var = np.asarray(samples_var)
samples_var_1=samples_var.reshape(-1,2)#虚拟隐变量
Vx_separate_var_1 = np.dot(samples_var_1,W.transpose())
Vx_z_separate_var_1= s_scale.inverse_transform(Vx_separate_var_1)#虚拟高维变量

#GPR回归
y1=np.array(dataset["Target"]).reshape(-1,1)
x1=np.array(dataset.iloc[:,2:65])
x1=tf.convert_to_tensor(x1, dtype=default_float())
y1=tf.convert_to_tensor(y1, dtype=default_float())
model = gpflow.models.GPR(
    (x1, y1),
    kernel=gpflow.kernels.SquaredExponential()
)
m=model
opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss,m.trainable_variables)
xnew_var=Vx_z_separate_var
mean_var,var_1=model.predict_f(xnew_var)
y=np.array(y)
mean_var=np.array(mean_var).flatten()
Y=np.concatenate((y,mean_var))
x1=np.array(x1)
Vx_z_separate_var=np.array(Vx_z_separate_var)
X=np.concatenate((x1,Vx_z_separate_var),axis=0)