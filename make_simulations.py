import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ[
    'XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import haiku as hk
from tqdm import tqdm
from jax.experimental.ode import odeint
from jaxpm.painting import cic_paint, cic_read, compensate_cic
from jaxpm.pm import linear_field, lpt, make_ode_fn, pm_forces, make_neural_ode_fn, pgd_correction
from jaxpm.kernels import fftk, gradient_kernel, laplace_kernel, longrange_kernel
from jaxpm.nn import NeuralSplineFourierFilter
from jaxpm.utils import power_spectrum
import jax_cosmo as jc
import numpyro 
import optax
import time
from functools import partial

flags.DEFINE_string("filename", "/data/CAMELS/Sims/PM_sims/lambda1_01/output_",
                    "Output filename")
flags.DEFINE_string("snapshots", "/local/home/dl264294/jaxpm-paper/notebooks/snapshots.params", "Scale factor of the napshot to use during the simultaions")
flags.DEFINE_string("correction_params_NN", "/local/home/dl264294/jaxpm-paper/notebooks/correction_params/camels_25_64_CV_0_lambda1_01.params", "Correction parameter files for NN")
flags.DEFINE_string("correction_params_PGD", "/local/home/dl264294/jaxpm-paper/notebooks/correction_params/camels_25_64_pkloss_PGD_CV_0.params", "Correction parameter files for PGD")
flags.DEFINE_float("Omega_m", 0.3 - 0.049, "Fiducial CDM and baryonic fraction")
flags.DEFINE_float("Omega_b",0.049, "Fiducial baryonic matter fraction")
flags.DEFINE_float("sigma8", 0.8, "Fiducial sigma_8 value")
flags.DEFINE_float("n_s", 0.9624, "Fiducial n_s value")
flags.DEFINE_float("h", 0.6711, "Fiducial Hubble constant value")
flags.DEFINE_integer("mesh_shape", 128,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_float("box_size", 200.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_integer("nsims", 1000, "Number simulations to generate.")



FLAGS = flags.FLAGS



@jax.jit
def compute_pk(seeds):
  k = jnp.logspace(-4, 2, 256)
  #Create a simple Planck15 cosmology
  cosmo = jc.Planck15(Omega_c= FLAGS.Omega_m - FLAGS.Omega_b, Omega_b=FLAGS.Omega_b, n_s=FLAGS.n_s, h=FLAGS.h, sigma8=FLAGS.sigma8)
  pk = jc.power.linear_matter_power(cosmo, k)
  pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)
  # Create some initial conditions
  print('Create initial conditions')
  initial_conditions = linear_field([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape], [FLAGS.box_size,FLAGS.box_size,FLAGS.box_size], pk_fn, seed=jax.random.PRNGKey(seeds))
  # Create particles
  particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in [FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]]),axis=-1).reshape([-1,3])
  snapshots = pickle.load(open(FLAGS.snapshots, "rb"))
  cosmo = jc.Planck15(Omega_c= FLAGS.Omega_m - FLAGS.Omega_b, Omega_b=FLAGS.Omega_b, n_s=FLAGS.n_s, h=FLAGS.h, sigma8=FLAGS.sigma8)
  dx, p, f = lpt(cosmo, initial_conditions, particles,snapshots[0])
  # Run the Nbody
  resi = odeint(make_ode_fn([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), [particles+dx, p], snapshots[::2], cosmo, rtol=1e-5, atol=1e-5)
  model = hk.without_apply_rng(hk.transform(lambda x,a : NeuralSplineFourierFilter(n_knots=16, latent_size=32)(x,a)))
  res = odeint(make_neural_ode_fn(model,[FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), [particles+dx, p], snapshots[::2], cosmo, pickle.load(open(FLAGS.correction_params_NN, "rb")), rtol=1e-5, atol=1e-5)
  print('Simulations done')
  # Let's compute the power spectrum
  k, pk_i = power_spectrum( compensate_cic(cic_paint(jnp.zeros([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), resi[0][-1])),
      boxsize=np.array([FLAGS.box_size,FLAGS.box_size,FLAGS.box_size]),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size)
  k, pk_nn = power_spectrum(compensate_cic(cic_paint(jnp.zeros([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), res[0][-1])),
      boxsize=np.array([FLAGS.box_size,FLAGS.box_size,FLAGS.box_size]),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size)
  params_PGD=pickle.load(open(FLAGS.correction_params_PGD, "rb"))
  fac_PGD=FLAGS.box_size/FLAGS.mesh_shape*(64/25)  #64/25= nc/box_size of the original Camels simulations used to train PGD
  params_PGD=[params_PGD[0],params_PGD[1]*fac_PGD,params_PGD[2]*fac_PGD]
  k, pk_pgd = power_spectrum(
      (cic_paint(jnp.zeros([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), resi[0][-1]+pgd_correction(resi[0][-1], [FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape],cosmo,params_PGD))),
      boxsize=np.array([FLAGS.box_size,FLAGS.box_size,FLAGS.box_size]),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size) 
  return k, pk_nn, pk_pgd, pk_i

def main(_):
    t = time.time()
    for i in range(518,FLAGS.nsims):
        k, pk_NN, pk_pgd, pk_i= compute_pk(int(time.time()))
        pickle.dump(
              {    'k':k,
                    'pk_NN':pk_NN,
                    'pk_pgd':pk_pgd,
                    'pk_i':pk_i
              },
              open(FLAGS.filename + 'sims_%d' % i+'.pkl', "wb"))
        print("iter", i, "took", time.time() - t)
        

if __name__ == "__main__":
  app.run(main)