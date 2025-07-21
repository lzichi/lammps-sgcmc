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

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::FixSemiGrandCanonicalMCSectorKokkos(LAMMPS *lmp, int narg, char **arg) :
 FixSemiGrandCanonicalMCSector(lmp, narg, arg), rand_pool(seed), rand_pool_local(seed + comm->me)
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

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::init()
{
    FixSemiGrandCanonicalMCSector::init();
    
    // TODO: how to properly initialized the kokkos k_speciesCounts
    Kokkos::resize(k_speciesCounts, atomKK->ntypes + 1);
    Kokkos::deep_copy(k_speciesCounts, speciesCounts);
    d_speciesCounts = k_speciesCounts.template view<DeviceType>();
    h_speciesCounts = k_speciesCounts.h_vew;

    // TODO: how to properly initialize device only views

    // rsec filled in by sectoring() called inside init()
    auto k_rsec = DAT::tdual_int_1d("fix:rsec", 3);
    auto h_rsec = k_rsec.h_vew;

    for (i = 0; i < 3; i++) {
        h_rsec(i) = rsec[i];
    }

    k_rsec.template modify<LMPHostType>();
    k_rsec.template sync<DeviceType>();
    d_rsec = k_rsec.template view<DeviceType>();

    // d_num_atoms_per_sector = typename AT::t_int_1d("fix:num_atoms_in_sector", nsectors);
    // d_atoms_in_sector = typename AT::t_int_1d("fix:atoms_in_sector", atom->nlocal);
    // d_stack_foot = typename AT::t_int_1d("fix:stack_foot", nsectors);
    // d_forward_stacks = typename AT::t_int_1d("fix:forward_stacks", atom->nlocal);

}

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::init_list(int id, NeighList *ptr)
{
    FixSemiGrandCanonicalMCSectorSector::init_list(id, ptr);
    k_neighborList = static_cast<NeighListKokkos<DeviceType>*>(ptr);
}
/*********************************************************************
 *********************************************************************/

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::setup(int vflag)
{
    // TODO: where should this actually go
    x = atomKK->k_x.view<DeviceType>();
    f = atomKK->k_f.view<DeviceType>();
    type = atomKK->k_type.view<DeviceType>();
    mask = atomKK->k_mask.view<DeviceType>();
    ntypes = atomKK->ntypes;

    d_numneigh = k_neighborList->d_numneigh;
    d_neighbors = k_neighborList->d_neighbors;
    d_ilist = k_neighborList->d_neighors;
    inum = list->inum;
    nlocal = atom->nlocal;
}

/*********************************************************************
 * Lets the fix report one of its internal state variables to LAMMPS.
 *********************************************************************/
 double FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::compute_vector(int index)
 {
    if (index == 0) return nAcceptedSwaps;
    if (index == 1) return nRejectedSwaps;
    index -=1;
    int totalAtoms = 0;
    
    for (int i = 0; i < (int)h_speciesCounts.size_t; i++)
        totalAtoms += h_speciesCounts(i);

    if (index <= atom->ntypes)
        return static_cast<double>(h_speciesCounts(index)) / (totalAtoms > 0 ? totalAtoms : 1);

    return 0.0;
 }

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixSemiGrandCanonicalMCSectorKokkos::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                            DAT::tdual_xfloat_1d &buf,
                                                            int /*pbc_flag*/, int * /*pbc*/)
{
    d_sendlist = k_sendlist.view<DeviceType>();
    v_buf = buf.view<DeviceType>();
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixSemiGrandCanonicalMCSectorPackForwardComm>(0, n), *this);
    return n;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::operator()(TagFixSemiGrandCanonicalMCSectorPackForwardComm, const int &i) const {
    int j = d_sendlist(i);
    v_buf[i] =  type[j];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos::unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_xfloat_1d &buf)
{
    first = first_in;
    v_buf = buf.view<DeviceType>();
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixSemiGrandCanonicalMCSectorUnpackForwardComm>(0,n), *this);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::operator()(TagFixSemiGrandCanonicalMCSectorUnpackForwardComm, const int &i) const {
    type[i + first] = (int)v_buf[i];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
                                                                 int /*pbc_flag*/, int * /*pbc*/)
{
    // TODO: call sync ?
    int i,j;

    for (i = 0; i < n; i++) {
        j = list[i];
        buf[i] = atom->type[j];
    }
    return n;
}            

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf);
{
    // TODO: do i need to sync or modify host?
    for (int i = 0; i < n; i++) {
        atom->type[i + first] = buf[i];
    }
}
                                 
/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
{
    // TODO: do i need to sync host?
    int i, m, last;
    m = 0;
    last = last + n;
    for (i = first; i < last; i++) buf[m++] = atom->type[i];
    return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
{
    // TODO: do i need to sync and modify the host?

    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
        j = list[i];
        atom->type += buf[m++];
    }
}

/* ---------------------------------------------------------------------- */

// template<class DeviceType>
// void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::doMC() {

//     // Reset counters
//     int nAcceptedSwapsLocal = 0;
//     int nRejectedSwapsLocal = 0;

//     int oldSpecies, newSpecies;

//     // Local and global change in number of atoms of each species
//     // TODO: should this be a dual view, might need to make a member
//     // variable if i want to use operators()
//     // typename AT::t_int_1d d_deltaN("d_deltaN", atom->ntypes + 1);
//     // typename AT::t_int_1d d_deltaNGlobal("d_deltaNGlobal", atom->ntypes + 1);

//     std::vector<int> deltaN(atom->ntypes+1, 0);         //< Local change in number of atoms of each species.
//     std::vector<int> deltaNGlobal(atom->ntypes+1, 0);   //< Global change in number of atoms of each species.

//     for (int i = 0; i < nsectors; i++) {

//         int nDice = (int)(swap_fraction * numFixAtomsLocal / numSamplingWindowMoves);

//         int largestnDice;
//         MPI_Allreduce(&nDice, &largestnDice, 1, MPI_INT, MPI_MAX, world);

//         // The probability to do one swap step.
//         double diceProbability = (double)nDice / (double)largestnDice;

//         // Inner MC loop that swaps atom types
//         for (int j = 0; j < largestDice; j++) {

//             double deltaE = 0.0;
//             std::fill(deltaN.begin(), deltaN.end(), 0.0);
//             int selectedAtom = -1;
//             int selectedAtomNL = -1;


//         }
//     }

//     // MPI sum total number of accepted/rejected swaps.
//     MPI_Allreduce(&nAcceptedSwapsLocal, &nAcceptedSwaps, 1, MPI_INT, MPI_SUM, world);
//     MPI_ALLreduce(&nRejectedSwapsLocal, &nRejectedSwaps, 1, MPI_INT, MPI_SUM, world);

//     if (kappa == 0.0 && !serialMode) {
//         // TODO: should these types be kokkos??
//         // TODO: can you do an MPI reduce over kokkos views
//         // TODO: can you parallelize this without a race condition 
//         // TODO: should localSpeciesCounts be kokkos view or dual view maybe
//         for( int i = 0; i < nlocal; i++) {
//             if(mask(i) & groupbit)
//                 localSpeciesCounts[type[i]]++; // TODO: is this the same thing as what is in fix_sgcmc
//         }

//         MPI_ALLreduce(localSpeciesCounts.data(), speciesCounts.data(), localSpeciesCounts.size(), MPI_INT, MPI_SUM, world);
//     }
// }

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::computeEnergyChangeEatom(int flipAtom, int oldSpecies, int newSpecies) {

    double Eold, Enew, deltaE;

    Eold = force->pair->compute_atomic_energy(flipAtom, neighborList);

    int jnum = neighborList->numneigh[flipAtom];

    Kokkos::parallel_reduce("computeEnergyChangeEatom:Eold", jnum, KOKKOS_LAMBDA(const int& jj, double& E_par) {
        E_par += force->pair->compute_atom(jj, neighborList);
    }, Eold);

    atom->type[flipAtom] = newSpecies;
    atomKK.modify_host(); // TODO: is this correct?

    Enew = force->pair->compute_atomic_energy(flipAtom, neighborList);

    Kokkos::parallel_reduce("computeEnergyChangeEatom:Eold", jnum, KOKKOS_LAMBDA(const int& jj, double& E_par) {
        E_par += force->pair->compute_atom(jj, neighborList);
    }, Enew);

    atom->type[flipAtom] = oldSpecies;
    atomKK.modify_host();

    deltaE = Enew - Eold;

    return deltaE;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::pre_neighbor() {

    // TODO: ideally find a  way to parallelize this instead of copying 
    // info from base class

    // if(nlocal_max < atom->nlocal) {
    //     nlocal_max = atom->nlocal;
    //     d_atoms_in_sector = typename AT::t_int_1d("fix:atoms_in_sector", nlocal_max);
    //     d_forward_stacks = typename AT::t_int_1d("fix:forward_stacks", nlocal_max);
    // }

    // for (int j = 0; j < nsectors; j++) {
    //     d_stack_foot(j) = -1;
    // }

    // int nseci;

    FixSemiGrandCanonicalMCSector::per_neighbor();

    auto k_num_atoms_per_sector = DAT::tdual_int_1d("fix:num_atoms_per_sector", nsectors);
    auto k_atoms_in_sector = DAT::tdual_int_1d("fix:atoms_in_sector", atoms->nlocal);

    auto h_num_atoms_per_sector = k_num_atoms_per_sector.h_view;
    auto h_atoms_in_sector = k_atoms_in_sector.h_view;

    for (int i = 0; i < nsectors; i++) {
        h_num_atoms_per_sector(i) = num_atoms_in_sector[i];
    }

    for (int i = 0; i < atom->nlocal; i++) {
        h_atoms_in_sector(i) = atoms_in_sector[i];
    }

    // TODO: why does EAM modify and sync the k_ types?
    k_num_atoms_per_sector.template modify<LMPHostType>();
    k_num_atoms_per_sector.template sync<DeviceType>();
    k_atoms_in_sector.template modify<LMPHostType>();
    k_atoms_in_sector.template sync<DeviceType>();

    d_num_atoms_per_sector = k_num_atoms_per_sector.template view<DeviceType>();
    d_atoms_in_sector = k_atoms_in_sector.template view<DeviceType>();

}


