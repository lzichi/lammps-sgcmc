/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 * Parallel Monte-Carlo code for the semi-grandcanonical ensemble (SGC)
 * and the variance-constrained semi-grandcanonical ensemble (VC-SGC).
 *
 * See Sadigh et al., Phys. Rev. B 85, 184203 (2012) for a
 * description of the algorithm.
 *
 * Code author: Alexander Stukowski (stukowski@mm.tu-darmstadt.de)
 *
 * Updates to use the sectoring method for parallelism: Aidan Thompson, SNL,
 * Laura Zichi, SNL and Axel Kohlmeyer, Temple U
------------------------------------------------------------------------- */

#include "fix_sgcmcs.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "integrate.h"
#include "kspace.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "universe.h"
#include "update.h"
#include "memory.h"

#include "random_park.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/*********************************************************************
 * Constructs the fix object and parses the input parameters
 * that control the Monte Carlo routine.
 *********************************************************************/
FixSemiGrandCanonicalMCSector::FixSemiGrandCanonicalMCSector(LAMMPS *_lmp, int narg, char **arg) :
    Fix(_lmp, narg, arg), random(nullptr), localRandom(nullptr), neighborList(nullptr),
    compute_pe(nullptr), rsec(nullptr), atoms_in_sector(nullptr), num_atoms_per_sector(nullptr),
    ids(nullptr)
{
  scalar_flag = 0;
  vector_flag = 1;
  extvector = 0;
  global_freq = 1;

  // Specifies the number of output fields this fix produces for thermo output.
  // It calculates the
  //    - Number of accepted trial moves
  //    - Number of rejected trial moves
  //    - Atom counts for each species.
  size_vector = 2 + atom->ntypes;

  nAcceptedSwaps = 0;
  nRejectedSwaps = 0;
  kappa = 0;

  // Determine if using parallel sectoring algorithm or serial run
  //sector_flag = (comm->nprocs > 1) ? 1 : 0;
  sector_flag = 1; // for now do sectoring in serial

  if (domain->triclinic)
    error->all(FLERR, "Fix sgcmcs does not support non-orthogonal simulation boxes.");

  if (narg < 6) utils::missing_cmd_args(FLERR, "fix sgcmcs", error);

  // Parse the number of MD timesteps to do between MC.
  nevery_mdsteps = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery_mdsteps <= 0) error->all(FLERR, "Invalid number of MD timesteps {}", nevery_mdsteps);
  if (comm->me == 0) utils::logmesg(lmp, "  SGC - Number of MD timesteps: {}\n", nevery_mdsteps);

  // Parse the fraction of atoms swaps attempted during each cycle.
  swap_fraction = utils::numeric(FLERR, arg[4], false, lmp);
  if ((swap_fraction < 0.0) || (swap_fraction > 1.0))
    error->all(FLERR, "Invalid fraction {} of swap atoms", swap_fraction);
  if (comm->me == 0) utils::logmesg(lmp, "  SGC - Fraction of swap atoms: {}\n", swap_fraction);

  // Parse temperature for MC.
  double temperature = utils::numeric(FLERR, arg[5], false, lmp);
  if (temperature <= 0) error->all(FLERR, "Temperature {} invalid", temperature);
  if (comm->me == 0) utils::logmesg(lmp, "  SGC - Temperature: {}\n", temperature);
  beta = 1.0 / (force->boltz * temperature);

  // Parse chemical potentials.
  int iarg = 6;
  deltamu.resize(atom->ntypes + 1);
  deltamu[0] = 0.0;
  deltamu[1] = 0.0;
  if (atom->ntypes < 2)
    error->all(FLERR, "Fix sgcmcs can only be used in simulations with at least two atom types.");
  for (int i = 2; i <= atom->ntypes; i++, iarg++) {
    if (iarg >= narg) error->all(FLERR, "Too few chemical potentials specified");
    deltamu[i] = utils::numeric(FLERR, arg[iarg], false, lmp);
    if (comm->me == 0)
      utils::logmesg(lmp, "  SGC - Chemical potential of species {}: {}\n", i, deltamu[i]);
  }

  // Default values for optional parameters (where applicable).
  seed = 324234;
  atomicenergyflag = 0;

  // Parse extra/optional parameters
  while (iarg < narg) {

    if (strcmp(arg[iarg], "randseed") == 0) {
      // Random number seed.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix sgcmcs randseed", error);
      seed = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (seed <= 0) error->all(FLERR, "Random number seed {} must be positive", seed);
      if (comm->me == 0) utils::logmesg(lmp, "  SGC - Random number seed: {}\n", seed);
      iarg += 2;

    } else if (strcmp(arg[iarg], "variance") == 0) {
      // Parse parameters for variance constraint ensemble.
      if (iarg + 1 + atom->ntypes > narg)
        utils::missing_cmd_args(FLERR, "fix sgcmcs variance", error);
      iarg++;

      kappa = utils::numeric(FLERR, arg[iarg], false, lmp);
      if (kappa < 0) error->all(FLERR, "Variance constraint parameter must not be negative.");
      if (comm->me == 0) utils::logmesg(lmp, "  SGC - Kappa: {}\n", kappa);
      iarg++;

      targetConcentration.resize(atom->ntypes + 1);
      targetConcentration[0] = 1.0;
      targetConcentration[1] = 1.0;
      for (int i = 2; i <= atom->ntypes; i++, iarg++) {
        targetConcentration[i] = utils::numeric(FLERR, arg[iarg], false, lmp);
        targetConcentration[1] -= targetConcentration[i];
      }
      for (int i = 1; i <= atom->ntypes; i++) {
        if ((targetConcentration[i] < 0.0) || (targetConcentration[i] > 1.0))
          error->all(FLERR, "Target concentration {} for species {} is out of range",
                     targetConcentration[i], i);
        if (comm->me == 0)
          utils::logmesg(lmp, "  SGC - Target concentration of species {}: {}\n", i,
                         targetConcentration[i]);
      }
    } else if (strcmp(arg[iarg],"atomic/energy") == 0) {
      if (iarg+2 > narg) utils::missing_cmd_args(FLERR, fmt::format("fix {} atomic/energy", style), error);
      atomicenergyflag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else {
      error->all(FLERR, "Unknown fix sgcmc keyword: {}", arg[iarg]);
    }
  }

  // Initialize random number generators.
  random = new RanPark(lmp, seed);
  localRandom = new RanPark(lmp, seed + universe->me);
}

/*********************************************************************
 * Destructor. Cleans up the random number generators.
 *********************************************************************/
FixSemiGrandCanonicalMCSector::~FixSemiGrandCanonicalMCSector()
{
  memory->destroy(rsec);
  memory->destroy(num_atoms_per_sector);
  memory->destroy(atoms_in_sector);
  memory->destroy(ids);
  delete random;
  delete localRandom;
}

/*********************************************************************
 * The return value of this method specifies at which points the
 * fix is invoked during the simulation.
 *********************************************************************/
int FixSemiGrandCanonicalMCSector::setmask()
{
  // We want the MC routine to be called in between the MD steps.
  // We need the electron densities for each atom, so after the
  // EAM potential has computed them in the force routine is a good
  // time to invoke the MC routine.
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= PRE_NEIGHBOR;

  return mask;
}

/*********************************************************************
 * This gets called by the system before the simulation starts.
 *********************************************************************/
void FixSemiGrandCanonicalMCSector::init()
{

  // Let LAMMPS know the number of data values per atom to transfer in MPI communication.
  // Depends on what pair_style is used
  comm_forward = force->pair->comm_forward;
  comm_reverse = force->pair->comm_reverse;

  if (!atom->mass) error->all(FLERR, "Fix sgcmcs requires per atom type masses");
  if (atom->rmass_flag && (comm->me == 0))
    error->warning(FLERR, "Fix sgcmcs will use per atom type masses for velocity initialization");

  // Make sure the user has defined only one Semi-Grand Monte-Carlo fix.
  if (modify->get_fix_by_style("sgcmcs").size() > 1)
    error->all(FLERR, "More than one fix sgcmcs defined.");

  if (atomicenergyflag) {
    // Save a pointer to the EAM potential.
    if (comm->me == 0) utils::logmesg(lmp, "  SGC - Using atomic energy method for SGCMC\n");
    if (!force->pair->atomic_energy_enable) {
      error->all(FLERR, "SGC - Pair style does not support atomic energy method");
    }
    compute_pe = modify->get_compute_by_id("thermo_pe");
  } else {
    // // Save a pointer to the EAM potential.
    // pairEAM = dynamic_cast<PairEAM*>(force->pair);
    // if (!pairEAM) {

      if (comm->me == 0)
        utils::logmesg(lmp, "  SGC - Using naive total energy calculation for MC -> SLOW!\n");

      if (comm->nprocs > 1)
        error->all(FLERR, "Can not run fix sgcmcs with naive total energy calculation "
                   "and more than one MPI process.");

      // Get reference to a compute that will provide the total energy of the system.
      // This is needed by computeTotalEnergy().
      compute_pe = modify->get_compute_by_id("thermo_pe");
   // }
  }

  interactionRadius = force->pair->cutforce;
  if (comm->me == 0) utils::logmesg(lmp, "  SGC - Interaction radius: {}\n", interactionRadius);

  // This fix needs a full neighbor list.
  if (atomicenergyflag)
    // for atomic energy method, need ghost neighbors
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  else
    neighbor->add_request(this, NeighConst::REQ_FULL);

  // Count local number of atoms from each species.
  const int *type = atom->type;
  const int *mask = atom->mask;
  std::vector<int> localSpeciesCounts(atom->ntypes+1, 0);
  for (int i = 0; i < atom->nlocal; i++, ++type) {
    if (mask[i] & groupbit)
      localSpeciesCounts[*type]++;
  }

  // MPI sum to get global concentrations.
  speciesCounts.resize(atom->ntypes+1);
  MPI_Allreduce(localSpeciesCounts.data(), speciesCounts.data(), localSpeciesCounts.size(),
                MPI_INT, MPI_SUM, world);

  // setting the sector variables/lists
  nsectors = 0;
  nlocal_max = 0;
  memory->grow(rsec,3,"sgcmcs:rsec");
  memory->grow(atoms_in_sector, atom->nlocal, "sgcmcs:atoms_in_sector");

  ids_size = 0;

  // perform the sectoring operation
  if (sector_flag) sectoring();
  
  // init. size of stacking lists (sectoring)
  memory->grow(num_atoms_per_sector,nsectors,"sgcmcs:num_atoms_per_sector");
}

/*********************************************************************
 * Assigns the requested neighbor list to the fix.
 *********************************************************************/
void FixSemiGrandCanonicalMCSector::init_list(int /*id*/, NeighList *ptr)
{
  neighborList = ptr;

}

/*********************************************************************
 * Called after the pair_style force calculation during each timestep.
 * This method triggers the MC routine from time to time.
 *********************************************************************/
void FixSemiGrandCanonicalMCSector::post_force(int /*vflag*/)
{
  if ((update->ntimestep % nevery_mdsteps) == 0)
    doMC();
}

/*********************************************************************
 * This routine does one full MC step.
 *********************************************************************/
void FixSemiGrandCanonicalMCSector::doMC()
{
  // Get information about local ghost atoms from neighboring nodes
  // TODO: Question: decide on this "communicationStage" parameter
  // TODO: Question: do we need to do this communication?
//   communicationStage = 1; 
//   comm->forward_comm(this);

  const int *mask = atom->mask;

  // Reset counters.
  int nAcceptedSwapsLocal = 0;
  int nRejectedSwapsLocal = 0;

  int oldSpecies, newSpecies;
  std::vector<int> deltaN(atom->ntypes+1, 0);         //< Local change in number of atoms of each species.
  std::vector<int> deltaNGlobal(atom->ntypes+1, 0);   //< Global change in number of atoms of each species.

  numFixAtomsLocal = 0; // number of atoms that a processor owns

  for (int ii = 0; ii < neighborList->inum; ii++) {
    int i = neighborList->ilist[ii];
    if (mask[i] & groupbit) {
        numFixAtomsLocal++;
    }
  }

  int offset = 0;
  // loop through each sector and run MC
  for (int j_sector = 0; j_sector < nsectors; j_sector++) {

    // TODO: Question: keep it like this? or have each processor pick a number
    /// The number of times we want to swap an atom.
    int nDice = (int)(swap_fraction * numFixAtomsLocal / nsectors);

    // This number must be synchronized with the other nodes. We take the largest
    // of all nodes and skip trial moves later.
    int largestnDice;
    MPI_Allreduce(&nDice, &largestnDice, 1, MPI_INT, MPI_MAX, world);

    // The probability to do one swap step.
    double diceProbability = (double)nDice / (double)largestnDice;

    // Inner MC loop that swaps atom types.
    for (int j = 0; j < largestnDice; j++) {

      double deltaE = 0;
      std::fill(deltaN.begin(), deltaN.end(), 0);
      int selectedAtom = -1, selectedAtomNL = -1;

      // As already said above, we have to do swap steps only with a certain probability
      // to keep nodes in sync.
      if (localRandom->uniform() <= diceProbability) {

        // Choose a random atom from the pool of atoms that are inside the sampling window.
        int index = (int)(localRandom->uniform() * (double)num_atoms_per_sector[j_sector]);
        selectedAtomNL = atoms_in_sector[index + offset];

        // Get the real atom index.
        selectedAtom = neighborList->ilist[selectedAtomNL];
        oldSpecies = atom->type[selectedAtom];

        // Choose the new type for the swapping atom by random.
        if (atom->ntypes > 2) {
          // Use a random number to choose the new species if there are three or more atom types.
          newSpecies = (int)(localRandom->uniform() * (atom->ntypes-1)) + 1;
          if (newSpecies >= oldSpecies) newSpecies++;
        } else {
          // If there are only two atom types, then the decision is clear.
          newSpecies = (oldSpecies == 1) ? 2 : 1;
        }
        deltaN[oldSpecies] = -1;
        deltaN[newSpecies] = +1;

        // Compute the energy difference that swapping this atom would cost or gain.

        // Atomic energy method:
        if(atomicenergyflag) {
          deltaE = computeEnergyChangeEatom(selectedAtom, oldSpecies, newSpecies);
        // Slow generic method:
        } else {
            deltaE = computeEnergyChangeGeneric(selectedAtom, oldSpecies, newSpecies);
        }

        // Perform inner MC acceptance test.
        double dm = 0.0;
        if (!sector_flag && kappa != 0.0) {
          for (int i = 2; i <= atom->ntypes; i++)
            dm += (deltamu[i] + kappa / atom->natoms * (2.0 * speciesCounts[i] + deltaN[i])) * deltaN[i];
        } else {
          for (int i = 2; i <= atom->ntypes; i++)
            dm += deltamu[i] * deltaN[i];
        }
        double deltaB = -(deltaE + dm) * beta;
        if (deltaB < 0.0) {
          if (deltaB < log(localRandom->uniform())) {
            std::fill(deltaN.begin(), deltaN.end(), 0);
            selectedAtom = -1;
            deltaE = 0;
          }
        }
      }

      if (kappa != 0.0 && sector_flag) {

        // What follows is the second rejection test for the variance-constrained
        // semi-grandcanonical method.

        // MPI sum of total change in number of particles.
        MPI_Allreduce(deltaN.data(), deltaNGlobal.data(), deltaN.size(), MPI_INT, MPI_SUM, world);

        // Perform outer MC acceptance test.
        // This is done in sync by all processors.
        double A = 0.0;
        for (int i = 1; i <= atom->ntypes; i++) {
          A += deltaNGlobal[i] * deltaNGlobal[i];
          A += 2.0 * deltaNGlobal[i] * (speciesCounts[i] - (int)(targetConcentration[i] * atom->natoms));
        }
        double deltaB = -(kappa / atom->natoms) * A;
        if (deltaB < 0.0) {
          if (deltaB < log(random->uniform())) {
            std::fill(deltaN.begin(), deltaN.end(), 0);
            std::fill(deltaNGlobal.begin(), deltaNGlobal.end(), 0);
            selectedAtom = -1;
          }
        }

        // Update global species counters.
        for (int i = 1; i <= atom->ntypes; i++)
          speciesCounts[i] += deltaNGlobal[i];
      } else if (!sector_flag) {
        // Update the local species counters.
        for (int i = 1; i <= atom->ntypes; i++)
          speciesCounts[i] += deltaN[i];
      }

      // Make accepted atom swap permanent.
      if (selectedAtom >= 0) {
        if(atomicenergyflag) {
          flipAtomEatom(selectedAtom, oldSpecies, newSpecies);
        } else {
          flipAtomGeneric(selectedAtom, oldSpecies, newSpecies);
        }
        nAcceptedSwapsLocal++;
      } else {
        nRejectedSwapsLocal++;
      }

    }
    
    offset += num_atoms_per_sector[j_sector];

    // communicate ghost atom information to neighboring processors
    // before moving onto the next sector
    communicateTypes();
  }

  // MPI sum total number of accepted/rejected swaps.
  MPI_Allreduce(&nAcceptedSwapsLocal, &nAcceptedSwaps, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&nRejectedSwapsLocal, &nRejectedSwaps, 1, MPI_INT, MPI_SUM, world);

  // For (parallelized) semi-grandcanonical MC we have to determine the current concentrations now.
  // For the serial version and variance-constrained MC it has already been done in the loop.
  if (kappa == 0.0 && sector_flag) {
    const int *type = atom->type;
    std::vector<int> localSpeciesCounts(atom->ntypes+1, 0);
    for (int i = 0; i < atom->nlocal; i++, ++type) {
      if (mask[i] & groupbit)
        localSpeciesCounts[*type]++;
    }
    MPI_Allreduce(localSpeciesCounts.data(), speciesCounts.data(), localSpeciesCounts.size(), MPI_INT, MPI_SUM, world);
  }
}

/*********************************************************************
 * Transfers the locally changed electron densities and atom
 * types to the neighbors.
 *********************************************************************/
void FixSemiGrandCanonicalMCSector::communicateTypes()
{
  // Transfer changed atom types and electron densities of the real atoms to the ghost atoms.
  communicationStage = 3;
  comm->forward_comm(this);
}

/*********************************************************************
 * This is for MPI communication with neighbor nodes.
 *********************************************************************/
int FixSemiGrandCanonicalMCSector::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                               int * /*pbc*/)
{
  int m = 0;
  if (communicationStage == 3) {
    // Generic potential case:
    for (int i = 0; i < n; i++) {
    buf[m++] = atom->type[list[i]];
    }
  } 
  return m;
}

/*********************************************************************
 * This is for MPI communication with neighbor nodes.
 *********************************************************************/
void FixSemiGrandCanonicalMCSector::unpack_forward_comm(int n, int first, double* buf)
{
  if (communicationStage == 3) {
    int last = first + n;
    for (int i = first; i < last; i++, buf += 1) {
    atom->type[i] = (int)buf[0];
    }
  } 
}

/*********************************************************************
 * Calculates the change in energy that swapping the given atom would produce.
 * This routine is for the general case of an arbitrary potential and
 * IS VERY SLOW! It computes the total energies of the system for the unmodified state
 * and for the modified state and then returns the difference of both values.
 * This routine should only be used for debugging purposes.
 *
 * Parameters:
 *
 * flipAtom [in]
 *   This specifies the atom to be swapped. It's an index into the local list of atoms.
 *
 * oldSpecies [in]
 *   The current species of the atom before the routine is called.
 *
 * newSpecies [in]
 *   The new species of the atom. The atom's type is not changed by this method. It only computes the induced energy change.
 *
 * Return value:
 *   The expected change in total potential energy.
 *********************************************************************/
double FixSemiGrandCanonicalMCSector::computeEnergyChangeGeneric(int flipAtom, int oldSpecies, int newSpecies)
{
  // This routine is called even when no trial move is being performed during the
  // the current iteration to keep the parallel processors in sync. If no trial
  // move is performed then the energy is calculated twice for the same state of the system.
  if (flipAtom >= 0) {
    // Change system. Perform trial move.
    atom->type[flipAtom] = newSpecies;
  }
  // Transfer changed atom types of the real atoms to the ghost atoms.
  communicationStage = 3;
  comm->forward_comm(this);

  // Calculate new total energy.
  double newEnergy = computeTotalEnergy();

  // Undo trial move. Restore old system state.
  if (flipAtom >= 0) {
    atom->type[flipAtom] = oldSpecies;
  }
  // Transfer changed atom types of the real atoms to the ghost atoms.
  communicationStage = 3;
  comm->forward_comm(this);

  // Calculate old total energy.
  double oldEnergy = computeTotalEnergy();

  // Restore the correct electron densities and forces.
  update->integrate->setup_minimal(0);

  return newEnergy - oldEnergy;
}

/*********************************************************************
 * Lets LAMMPS calculate the total potential energy of the system.
 *********************************************************************/
double FixSemiGrandCanonicalMCSector::computeTotalEnergy()
{
  int eflag = 1;
  int vflag = 0;

  if (force->pair) force->pair->compute(eflag,vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) force->kspace->compute(eflag,vflag);

  update->eflag_global = update->ntimestep;
  return compute_pe->compute_scalar();
}

/*********************************************************************
 * Flips the type of one atom.
 * This routine is for the generic case.
 *
 * Parameters:
 *
 * flipAtom [in]
 *   This specifies the atom to be swapped. It's an index into the local list of atoms.
 *
 * oldSpecies [in]
 *   The current species of the atom before the routine is called.
 *
 * newSpecies [in]
 *   The new type to be assigned to the atom.
 *********************************************************************/
void FixSemiGrandCanonicalMCSector::flipAtomGeneric(int flipAtom, int oldSpecies, int newSpecies)
{
  atom->type[flipAtom] = newSpecies;

  // Rescale particle velocity vector to conserve kinetic energy.
  double vScaleFactor = sqrt(atom->mass[oldSpecies] / atom->mass[newSpecies]);
  atom->v[flipAtom][0] *= vScaleFactor;
  atom->v[flipAtom][1] *= vScaleFactor;
  atom->v[flipAtom][2] *= vScaleFactor;

}

/*********************************************************************
 * Calculates the change in energy that swapping the given
 * atom would produce. This routine uses a per-atom energy calculation
 *********************************************************************/

double FixSemiGrandCanonicalMCSector::computeEnergyChangeEatom(int flipAtom, int oldSpecies, int newSpecies)
{
  double Eold, Enew, deltaE;

  // Calculate old atomic energy of selected atom
  Eold = force->pair->compute_atomic_energy(flipAtom, neighborList);

  // calculate the old per-atom energy of neighbors

  int* jlist = neighborList->firstneigh[flipAtom];
  int jnum = neighborList->numneigh[flipAtom];

  if (jnum > ids_size) {
    ids_size = jnum;
    memory->grow(ids, ids_size, "sgcmcs:ids");
  }
  
  for(int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj];
    ids[jj] = j;
    Eold += force->pair->compute_atomic_energy(j, neighborList);
  }
  //Eold += force->pair->compute_atomic_energy_batch(ids, neighborList, jnum);

  // Calculate new per-atom energy of selected atom

  atom->type[flipAtom] = newSpecies;

  Enew = force->pair->compute_atomic_energy(flipAtom, neighborList);

  // calculate the new per-atom energy of neighbors

  for(int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj];
    Enew += force->pair->compute_atomic_energy(j, neighborList);
    ids[jj] = j;
  }
  //Enew += force->pair->compute_atomic_energy_batch(ids, neighborList, jnum);

  atom->type[flipAtom] = oldSpecies;

  deltaE = Enew - Eold;

  return deltaE;
}

/*********************************************************************
 * Flips the type of one atom.
 * This routine is for the per-atom energy case.
 *********************************************************************/

void FixSemiGrandCanonicalMCSector::flipAtomEatom(int flipAtom, int oldSpecies, int newSpecies)
{
  atom->type[flipAtom] = newSpecies;

  // Rescale particle velocity vector to conserve kinetic energy.
  double vScaleFactor = sqrt(atom->mass[oldSpecies] / atom->mass[newSpecies]);
  atom->v[flipAtom][0] *= vScaleFactor;
  atom->v[flipAtom][1] *= vScaleFactor;
  atom->v[flipAtom][2] *= vScaleFactor;

}

/*********************************************************************
 * Lets the fix report one of its internal state variables to LAMMPS.
 *********************************************************************/
double FixSemiGrandCanonicalMCSector::compute_vector(int index)
{
  if (index == 0) return nAcceptedSwaps;
  if (index == 1) return nRejectedSwaps;
  index -= 1;
  int totalAtoms = 0;
  for (int i = 0; i < (int)speciesCounts.size(); i++)
    totalAtoms += speciesCounts[i];
  if (index <= atom->ntypes)
    return (double)speciesCounts[index] / (totalAtoms > 0 ? totalAtoms : 1);
  return 0.0;
}

/*********************************************************************
 * Reports the memory usage of this fix to LAMMPS.
 *********************************************************************/
double FixSemiGrandCanonicalMCSector::memory_usage()
{
  // TODO: do this actually
  return 0.0;
}

/*********************************************************************
 *  divide each domain into 8 sectors
 *********************************************************************/

void FixSemiGrandCanonicalMCSector::sectoring()
{
  int sec[3];
  double sublo[3],subhi[3];

  if (domain->triclinic == 1){
     double* sublotmp = domain->sublo_lamda;
     double* subhitmp = domain->subhi_lamda;
     for (int dim = 0 ; dim < 3 ; dim++) {
       sublo[dim]=sublotmp[dim]*domain->boxhi[dim];
       subhi[dim]=subhitmp[dim]*domain->boxhi[dim];
     }
  }

  else {
     double* sublotmp = domain->sublo;
     double* subhitmp = domain->subhi;
     for (int dim = 0 ; dim < 3 ; dim++) {
       sublo[dim]=sublotmp[dim];
       subhi[dim]=subhitmp[dim];
     }
  }

  const double rsx = subhi[0] - sublo[0];
  const double rsy = subhi[1] - sublo[1];
  const double rsz = subhi[2] - sublo[2];

  // extract larger cutoff from pair_style

  double rv, cutoff;
  rv = cutoff = force->pair->cutforce;

  if (rv == 0.0)
   error->all(FLERR, Error::NOLASTLINE,
              "No suitable cutoff found for sectoring operation: rv = {}", rv);

  double rax = rsx/rv;
  double ray = rsy/rv;
  double raz = rsz/rv;

  sec[0] = 1;
  sec[1] = 1;
  sec[2] = 1;
  if (rax >= 2.0) sec[0] = 2;
  if (ray >= 2.0) sec[1] = 2;
  if (raz >= 2.0) sec[2] = 2;

  nsectors = sec[0]*sec[1]*sec[2];

  if (sector_flag && (nsectors != 8))
    error->all(FLERR, Error::NOLASTLINE,
               "Illegal sectoring operation resulting in {} sectors instead of 8", nsectors);

  rsec[0] = rsx;
  rsec[1] = rsy;
  rsec[2] = rsz;
  if (sec[0] == 2) rsec[0] = rsx/2.0;
  if (sec[1] == 2) rsec[1] = rsy/2.0;
  if (sec[2] == 2) rsec[2] = rsz/2.0;

}

/*********************************************************************
 *  define sector for an atom at a position x[i]
 *********************************************************************/

int FixSemiGrandCanonicalMCSector::coords2sector(double *x)
{
  int nseci;
  int seci[3];
  double sublo[3];
  double* sublotmp = domain->sublo;
  for (int dim = 0 ; dim<3 ; dim++) {
    sublo[dim]=sublotmp[dim];
  }

  seci[0] = x[0] > (sublo[0] + rsec[0]);
  seci[1] = x[1] > (sublo[1] + rsec[1]);
  seci[2] = x[2] > (sublo[2] + rsec[2]);

  nseci = (seci[0] + 2*seci[1] + 4*seci[2]);

  return nseci;
}

/*********************************************************************
 *  setup pre_neighbor()
 *********************************************************************/

void FixSemiGrandCanonicalMCSector::setup_pre_neighbor()
{
  pre_neighbor();
}

/*********************************************************************
 *  store in two linked lists the advance order of the atoms for
 *  parallelism with the sectoring method
 *********************************************************************/

void FixSemiGrandCanonicalMCSector::pre_neighbor()
{

  double **x = atom->x;
  int nlocal = atom->nlocal;
  const int *mask = atom->mask;

  int *stack_foot;
  int *forward_stacks;

  memory->create(forward_stacks,nlocal,"sgcmcs:forward_stacks");
  memory->create(stack_foot,nsectors,"sgcmcs:stack_foot");

  if (nlocal_max < nlocal) {                    // grow linked lists if necessary
    nlocal_max = nlocal;
    memory->grow(atoms_in_sector,nlocal_max,"sgcmcs:atoms_in_sector");
  }
  for (int j = 0; j < nsectors; j++) {
    stack_foot[j] = -1;
  }
  int nseci;

  for (int j = nsectors-1; j >= 0; j--) {       // stacking forward order
    int num_atoms = 0;
    for (int i = nlocal-1; i >= 0; i--) {
      nseci = coords2sector(x[i]);
      if (j != nseci) continue;
      forward_stacks[i] = stack_foot[j];
      stack_foot[j] = i;
      num_atoms += 1;
    }
    num_atoms_per_sector[j] = num_atoms;
  }
  int index = 0;
  for (int j = 0; j < nsectors; j++) {          // store ids for each sector in order
    int ii = stack_foot[j];
    while (ii >= 0) {
        if(mask[ii] & groupbit) {
            atoms_in_sector[index] = ii;
            ii = forward_stacks[ii];
            index += 1;
        }
    }
  }

  memory->destroy(forward_stacks);
  memory->destroy(stack_foot);

}
