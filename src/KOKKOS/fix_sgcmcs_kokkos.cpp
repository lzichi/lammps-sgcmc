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
    datamask_read = X_MASK | TYPE_MASK;
    datamask_modify = TYPE_MASK;

    nmax = 0;
    maxj = 0;

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

    cutoff = force->pair->cutforce;

}

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::post_force(int /*vflag*/)
{

    if((update->ntimestep % nevery_mdsteps) == 0) {
        filter_neighbors();

        // run the MC
        doMC();
    }
}

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::filter_neighbors() 
{
    atomKK->sync(execution_space, datamask_read);

    x = atomKK->k_x.view<DeviceType>();
    type = atomKK->k_type.view<DeviceType>();

    NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(neighborList);

    // allocate views as necessary
    int temp_nmax = atom->nlocal + atom->nghost;
    if (temp_nmax > nmax) {
        nmax = temp_nmax;
        k_numneigh_short = DAT::tdual_int_1d("fix:numneigh", nmax);
        k_ilist_short = DAT::tdual_int_1d("fix:ilist", nmax);
    }

    int temp_maxj = k_list->d_neighbors.extent(1);
    if (temp_maxj > maxj) {
        maxj = temp_maxj;
        k_neighbors_short = DAT::tdual_int_2d("fix:neighbors", nmax, maxj);
    }

    d_numneigh_short = k_numneigh_short.template view<DeviceType>();
    d_neighbors_short = k_neighbors_short.template view<DeviceType>();
    d_ilist_short = k_ilist_short.template view<DeviceType>();

    d_ilist = k_list->d_ilist; 
    d_neighbors = k_list->d_neighbors;
    d_numneigh = k_list->d_numneigh;

    h_numneigh_short = k_numneigh_short.h_view;
    h_neighbors_short = k_neighbors_short.h_view;
    h_ilist_short = k_ilist_short.h_view;

    copymode = 1;
    
    Kokkos::parallel_for("fix:filter_neighbors", Kokkos::RangePolicy<DeviceType, TagFixSemiGrandCanonicalMCSectorFilterNeigh>(0, nmax), *this);
    copymode = 0;

    k_numneigh_short.template modify<DeviceType>();
    k_neighbors_short.template modify<DeviceType>();
    k_ilist_short.template modify<DeviceType>();

    k_numneigh_short.template sync<LMPHostType>();
    k_neighbors_short.template sync<LMPHostType>();
    k_ilist_short.template sync<LMPHostType>();
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::operator()(TagFixSemiGrandCanonicalMCSectorFilterNeigh, const int &ii) const{
    const int i = d_ilist[ii];
    d_ilist_short[ii] = i;
    const X_FLOAT xtmp = x(i, 0);
    const X_FLOAT ytmp = x(i, 1);
    const X_FLOAT ztmp = x(i, 2);
    
    const int jnum = d_numneigh[i];
    
    int inside = 0;
    for (int jj = 0; jj < jnum; jj++) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;

        const X_FLOAT delx = xtmp - x(j, 0);
        const X_FLOAT dely = ytmp - x(j, 1);
        const X_FLOAT delz = ztmp - x(j, 2);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        // if (rsq < cutoff*cutoff) {
        //     d_neighbors_short(i, inside) = j;
        //     inside++;
        // }
        d_neighbors_short(i, inside) = j;
        inside++;
    }
    d_numneigh_short(i) = inside;
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
  const int *mask = atom->mask;

  // Reset counters.
  int nAcceptedSwapsLocal = 0;
  int nRejectedSwapsLocal = 0;

  int oldSpecies, newSpecies;
  std::vector<int> deltaN(atom->ntypes+1, 0);         //< Local change in number of atoms of each species.
  std::vector<int> deltaNGlobal(atom->ntypes+1, 0);   //< Global change in number of atoms of each species.

  numFixAtomsLocal = 0; // number of atoms that a processor owns

//   TODO: Kokkos::parallel_for()
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
        selectedAtom = h_ilist_short[selectedAtomNL];
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


template<class DeviceType>
double FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::computeEnergyChangeEatom(int flipAtom, int oldSpecies, int newSpecies) 
{
  double Eold, Enew, deltaE;

  // Calculate old atomic energy of selected atom
  int jnum = h_numneigh_short[flipAtom];

  if (jnum > ids_size) {
    ids_size = jnum + 1;
    memory->grow(ids, ids_size, "sgcmcs:ids");
  }
  
  for(int jj = 0; jj < jnum; jj++) {
    int j = h_neighbors_short(flipAtom, jj);
    ids[jj] = j;
  }
  ids[jnum] = flipAtom;

  Eold = force->pair->compute_atomic_energy_batch(ids, neighborList, jnum);

  atom->type[flipAtom] = newSpecies;
  atomKK->modified(Host, TYPE_MASK);
  atomKK->sync(execution_space, TYPE_MASK);
  

  // calculate the new per-atom energy of neighbors

  for(int jj = 0; jj < jnum; jj++) {
    int j = h_neighbors_short(flipAtom, jj);
    ids[jj] = j;
  }
  ids[jnum] = flipAtom;

  Enew = force->pair->compute_atomic_energy_batch(ids, neighborList, jnum);

  atom->type[flipAtom] = oldSpecies;
  atomKK->modified(Host, TYPE_MASK);
  atomKK->sync(execution_space, TYPE_MASK);

  deltaE = Enew - Eold;

  return deltaE;
}

namespace LAMMPS_NS {
template class FixSemiGrandCanonicalMCSectorKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSemiGrandCanonicalMCSectorKokkos<LMPHostType>;
#endif
}
