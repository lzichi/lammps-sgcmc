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
 * Updates for integrtion into LAMMPS: Aidan Thompson, SNL and Axel Kohlmeyer, Temple U
 * Updates for Kokkos version: Laura Zichi, SNL
------------------------------------------------------------------------- */

#include "fix_sgcmcs_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "force.h"
#include "random_park.h"

using namespace LAMMPS_NS;

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::FixSemiGrandCanonicalMCSectorKokkos(LAMMPS *lmp, int narg, char **arg) :
 FixSemiGrandCanonicalMCSector(lmp, narg, arg)
{
    kokkosable = 1;
    atomKK = (AtomKokkos *) atom;
    execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

}

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::~FixSemiGrandCanonicalMCSectorKokkos()
{
    if (copymode) return;
}

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::init() 
{
    FixSemiGrandCanonicalMCSector::init();

    // adjust neighbor list request for KOKKOS

   // neighflag = lmp->kokkos->neighflag; // TODO: do i need this?
    auto request = neighbor->find_request(this);
    request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                            !std::is_same_v<DeviceType,LMPDeviceType>);
    request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);

}

// /* ---------------------------------------------------------------------- */

// template<class DeviceType>
// int FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
//                                                             DAT::tdual_xfloat_1d &buf,
//                                                             int /*pbc_flag*/, int * /*pbc*/)
// {
//     d_sendlist = k_sendlist.view<DeviceType>();
//     v_buf = buf.view<DeviceType>();
//     Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixSemiGrandCanonicalMCSectorPackForwardComm>(0, n), *this);
//     return n;
// }

// template<class DeviceType>
// KOKKOS_INLINE_FUNCTION
// void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::operator()(TagFixSemiGrandCanonicalMCSectorPackForwardComm, const int &i) const {
//     int j = d_sendlist(i);
//     v_buf[i] =  type[j];
// }

// /* ---------------------------------------------------------------------- */

// template<class DeviceType>
// void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_xfloat_1d &buf)
// {
//     first = first_in;
//     v_buf = buf.view<DeviceType>();
//     Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixSemiGrandCanonicalMCSectorUnpackForwardComm>(0,n), *this);
// }

// template<class DeviceType>
// KOKKOS_INLINE_FUNCTION
// void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::operator()(TagFixSemiGrandCanonicalMCSectorUnpackForwardComm, const int &i) const {
//     type[i + first] = (int)v_buf[i];
// }

// /* ---------------------------------------------------------------------- */

// template<class DeviceType>
// int FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
//                                                                  int /*pbc_flag*/, int * /*pbc*/)
// {
//     // TODO: call sync ?
//     int i,j;

//     for (i = 0; i < n; i++) {
//         j = list[i];
//         buf[i] = atom->type[j];
//     }
//     return n;
// }            

// /* ---------------------------------------------------------------------- */

// template<class DeviceType>
// void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
// {
//     // TODO: do i need to sync or modify host?
//     for (int i = 0; i < n; i++) {
//         atom->type[i + first] = buf[i];
//     }
// }
                                 
// /* ---------------------------------------------------------------------- */

// template<class DeviceType>
// int FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
// {
//     // TODO: do i need to sync host?
//     int i, m, last;
//     m = 0;
//     last = last + n;
//     for (i = first; i < last; i++) buf[m++] = atom->type[i];
//     return m;
// }

// /* ---------------------------------------------------------------------- */

// template<class DeviceType>
// void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
// {
//     // TODO: do i need to sync and modify the host?

//     int i, j, m;
//     m = 0;
//     for (i = 0; i < n; i++) {
//         j = list[i];
//         atom->type += buf[m++];
//     }
// }

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::doMC() 
{
    NeighListKokkos<DeviceType>* k_listneigh = static_cast<NeighListKokkos<DeviceType>*>(neighborList);
    d_ilist = k_listneigh->d_ilist;
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

//   Kokkos::parallel_for()
//   for (int ii = 0; ii < neighborList->inum; ii++) {
//     int i = d_ilist[ii];
//     if (mask[i] & groupbit) {
//         numFixAtomsLocal++;
//     }
//   }

  numFixAtomsLocal = atom->nlocal;

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
        //selectedAtom = d_ilist[selectedAtomNL];
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
          deltaE = computeEnergyChangeEatom(selectedAtomNL, oldSpecies, newSpecies);
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


template<class DeviceType>
double FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::computeEnergyChangeEatom(int flipAtom, int oldSpecies, int newSpecies) 
{
  double Eold, Enew, deltaE;

  NeighListKokkos<DeviceType>* k_listneigh = static_cast<NeighListKokkos<DeviceType>*>(neighborList);

  // Calculate old atomic energy of selected atom
  Eold = force->pair->compute_atomic_energy(flipAtom, neighborList);

  // calculate the old per-atom energy of neighbors

  d_neighbors = k_listneigh->d_neighbors;
  int jnum = k_listneigh->d_numneigh[flipAtom];

  if (jnum > ids_size) {
    ids_size = jnum;
    memory->grow(ids, ids_size, "sgcmcs:ids");
  }
  
  for(int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(flipAtom, jj);
    ids[jj] = j;
    //Eold += force->pair->compute_atomic_energy(j, neighborList);
  }
  Eold += force->pair->compute_atomic_energy_batch(ids, neighborList, jnum);

  // Calculate new per-atom energy of selected atom

  atom->type[flipAtom] = newSpecies;

  Enew = force->pair->compute_atomic_energy(flipAtom, neighborList);

  // calculate the new per-atom energy of neighbors

  for(int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(flipAtom, jj);
    //Enew += force->pair->compute_atomic_energy(j, neighborList);
    ids[jj] = j;
  }
  Enew += force->pair->compute_atomic_energy_batch(ids, neighborList, jnum);

  atom->type[flipAtom] = oldSpecies;

  deltaE = Enew - Eold;

  return deltaE;
}

namespace LAMMPS_NS {
template class FixSemiGrandCanonicalMCSectorKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSemiGrandCanonicalMCSectorKokkos<LMPHostType>;
#endif
}
