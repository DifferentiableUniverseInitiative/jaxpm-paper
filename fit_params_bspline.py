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
from jaxpm.pm import linear_field, lpt, make_ode_fn, pm_forces, make_neural_ode_fn
from jaxpm.kernels import fftk, gradient_kernel, laplace_kernel, longrange_kernel
from jaxpm.nn import NeuralSplineFourierFilter
from jaxpm.utils import power_spectrum
import jax_cosmo as jc
import numpyro 
import readgadget
import optax
from functools import partial


flags.DEFINE_string("filename", "camels_25_64_CV_0_lambda1_001.params", "Output filename")
flags.DEFINE_string("training_sims","/data/CAMELS/Sims/IllustrisTNG_DM/CV_0",
                    "Simulations used to train the NN")
flags.DEFINE_float("Omega_m", 0.3 - 0.049, "Fiducial CDM and baryonic fraction")
flags.DEFINE_float("Omega_b",0.049, "Fiducial baryonic matter fraction")
flags.DEFINE_float("sigma8", 0.8, "Fiducial sigma_8 value")
flags.DEFINE_float("n_s", 0.9624, "Fiducial n_s value")
flags.DEFINE_float("h", 0.6711, "Fiducial Hubble constant value")
flags.DEFINE_integer("mesh_shape", 64,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_float("box_size", 25.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_integer("niter", 500, "Number of iterations of loss fit")
flags.DEFINE_float("learning_rate", 0.01, "ADAM learning rate for the optim")
flags.DEFINE_boolean(
    "custom_weight", True,
    "Whether to apply a custom scale weighting to the loss function, or no weighting."
)
flags.DEFINE_float("lambda_2", 1., "Positive hyperparameters that allow us to tune the amount of regularisation given by the postion term")
flags.DEFINE_float("lambda_1", 0.01, "Positive hyperparameters that allow us to tune the amount of regularisation given by the  power spectrum term")



FLAGS = flags.FLAGS



@partial(jax.jit, static_argnames=['model'])
def loss_fn(params, cosmo, target_pos, target_vel, target_pk, scales, model):
  """
  Defines the loss function for the PGD parameters
  """

  # Step I: Compute the state vector
  res = odeint(make_neural_ode_fn(model,[FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), [target_pos[0], target_vel[0]], jnp.array(scales), cosmo, params, rtol=1e-5, atol=1e-5)
    
  # Step II:  Define a customized weight 
  distance = jnp.sum((res[0] - target_pos)**2, axis=-1)
  w = jnp.where(distance < 100, distance, 0.)
    
  # Step III: Painting and compute power spectrum
  k, pk = jax.vmap(lambda x: power_spectrum(
          (cic_paint(jnp.zeros([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), x)),
          boxsize=np.array([FLAGS.box_size] * 3),
          kmin=np.pi / FLAGS.box_size,
          dk=2 * np.pi / FLAGS.box_size))(res[0])
  # Step IV:  Compute loss
  if FLAGS.custom_weight:
    return FLAGS.lambda_2*jnp.mean(w)+FLAGS.lambda_1*jnp.mean(jnp.sum((pk/target_pk -1)**2,axis=-1))
  else:
    return FLAGS.lambda_1*jnp.mean(jnp.sum((pk/target_pk -1)**2,axis=-1))



@partial(jax.jit,  static_argnames=['model'])
def update( params, cosmo, target_pos, target_vel, target_pk, scales, model, opt_state):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, cosmo, target_pos, target_vel, target_pk, scales, model)
    optimizer = optax.adam(FLAGS.learning_rate)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

def main(_):
  #Create a simple Planck15 cosmology
  cosmo = jc.Planck15(Omega_c= FLAGS.Omega_m - FLAGS.Omega_b, Omega_b=FLAGS.Omega_b, n_s=FLAGS.n_s, h=FLAGS.h, sigma8=FLAGS.sigma8)  
  # Create some initial conditions
  print('Create initial conditions')
  init_cond=FLAGS.training_sims+'/ICs/ics'
  header   = readgadget.header(init_cond)
  BoxSize  = header.boxsize/1e3  #Mpc/h
  Nall     = header.nall         #Total number of particles
  Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
  Omega_m  = header.omega_m      #value of Omega_m
  Omega_l  = header.omega_l      #value of Omega_l
  h        = header.hubble       #value of h
  redshift = header.redshift     #redshift of the snapshot
  Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)
  ptype = [1] #dark matter is particle type 1
  ids_i = np.argsort(readgadget.read_block(init_cond, "ID  ", ptype)-1)  #IDs starting from 0
  pos_i = readgadget.read_block(init_cond, "POS ", ptype)[ids_i]/1e3     #positions in Mpc/h
  vel_i = readgadget.read_block(init_cond, "VEL ", ptype)[ids_i]         #peculiar velocities in km/s
  #Reordering data for simple reshaping
  re=256//FLAGS.mesh_shape #reshaping size
  pos_i = pos_i.reshape(re,re,re,FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
  vel_i = vel_i.reshape(re,re,re,FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
  pos_i = (pos_i/BoxSize*FLAGS.mesh_shape).reshape([256,256,256,3])[::re,::re,::re,:].reshape([-1,3])
  vel_i = (vel_i / 100 * (1./(1+redshift)) / BoxSize*FLAGS.mesh_shape).reshape([256,256,256,3])[::re,::re,::re,:].reshape([-1,3])
  a_i   = 1./(1+redshift)
  # Loading all the intermediate snapshots
  print('Loading the intermediate snapshots')
  scales = []
  poss = []
  vels = []
  for i in tqdm(range(34)):
    snapshot=FLAGS.training_sims+'/snap_%03d.hdf5'%i
    header   = readgadget.header(snapshot)
    redshift = header.redshift     #redshift of the snapshot
    h        = header.hubble       #value of h
    ptype = [1] #dark matter is particle type 1
    ids = np.argsort(readgadget.read_block(snapshot, "ID  ", ptype)-1)     #IDs starting from 0
    pos = readgadget.read_block(snapshot, "POS ", ptype)[ids] / 1e3        #positions in Mpc/h
    vel = readgadget.read_block(snapshot, "VEL ", ptype)[ids]              #peculiar velocities in km/s
    # Reordering data for simple reshaping
    pos = pos.reshape(re,re,re,FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
    vel = vel.reshape(re,re,re,FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
    
    pos = (pos / BoxSize *FLAGS.mesh_shape).reshape([256,256,256,3])[::re,::re,::re,:].reshape([-1,3])
    vel = (vel / 100 * (1./(1+redshift)) / BoxSize*FLAGS.mesh_shape).reshape([256,256,256,3])[::re,::re,::re,:].reshape([-1,3]) 
    scales.append((1./(1+redshift)))
    poss.append(pos)
    vels.append(vel)
  # Run the Nbody
  resi = odeint(make_ode_fn([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), [poss[0], vels[0]], jnp.array(scales), cosmo, rtol=1e-5, atol=1e-5)
  print('Simulation done')
  # Let's compute the target power spectrum
  ref_pos = jnp.stack(poss, axis=0)
  ref_vel = jnp.stack(vels, axis=0)
  ref_pk = jax.vmap(lambda x: power_spectrum(
      (cic_paint(jnp.zeros([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), x)),
      boxsize=np.array([FLAGS.box_size] * 3),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size)[1])(ref_pos)
  # Initialize NN params
  model = hk.without_apply_rng(hk.transform(lambda x,a : NeuralSplineFourierFilter(n_knots=16, latent_size=32)(x,a)))
  rng_seq = hk.PRNGSequence(1)
  params = model.init(next(rng_seq), jnp.zeros([FLAGS.mesh_shape,FLAGS.mesh_shape,FLAGS.mesh_shape]), jnp.ones([1]))
  losses = []
  optimizer = optax.adam(FLAGS.learning_rate)
  opt_state = optimizer.init(params)
  print('Starting fitting')
  for step in tqdm(range(FLAGS.niter)):
     l, params, opt_state = update(params, cosmo, ref_pos, ref_vel, ref_pk, scales, model, opt_state)
     losses.append(l)
  plt.plot(jnp.arange(0,FLAGS.niter),losses[:])
  plt.xlabel("niter")
  plt.ylabel("Loss")
  plt.savefig('Losses.png' )
  plt.close()
  pickle.dump(
          params
          , open(FLAGS.filename, "wb"))


if __name__ == "__main__":
  app.run(main)