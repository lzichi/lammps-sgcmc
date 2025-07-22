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
double FixSemiGrandCanonicalMCSectorKokkos<DeviceType>::computeEnergyChangeEatom(int flipAtom, int oldSpecies, int newSpecies) {

    double Eold, Enew, deltaE;

    atomKK->sync(force->pair->execution_space,force->pair->datamask_read);
    Eold = force->pair->compute_atomic_energy(flipAtom, neighborList);

    int jnum = neighborList->numneigh[flipAtom];

    Kokkos::parallel_reduce("computeEnergyChangeEatom:Eold", jnum, KOKKOS_LAMBDA(const int& jj, double& E_par) {
        E_par += force->pair->compute_atom(jj, neighborList);
    }, Eold);

    atom->type[flipAtom] = newSpecies;
    atomKK->modified(force->pair->execution_space,force->pair->datamask_modify);

    atomKK->sync(force->pair->execution_space,force->pair->datamask_read);
    Enew = force->pair->compute_atomic_energy(flipAtom, neighborList);

    Kokkos::parallel_reduce("computeEnergyChangeEatom:Eold", jnum, KOKKOS_LAMBDA(const int& jj, double& E_par) {
        E_par += force->pair->compute_atom(jj, neighborList);
    }, Enew);

    atom->type[flipAtom] = oldSpecies;
    atomKK->modified(force->pair->execution_space,force->pair->datamask_modify);

    deltaE = Enew - Eold;

    return deltaE;
}



