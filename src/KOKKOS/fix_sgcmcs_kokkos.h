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
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(sgcmcs/kk,FixSemiGrandCanonicalMCSectorKokkos<LMPDeviceType>);
FixStyle(sgcmcs/kk/device,FixSemiGrandCanonicalMCSectorKokkos<LMPDeviceType>);
FixStyle(sgcmcs/kk/host,FixSemiGrandCanonicalMCSectorKokkos<LMPHostType>);
// clang-format on
#else

#ifndef FIX_SGCMCS_KOKKOS_H
#define FIX_SGCMCS_KOKKOS_H

#include "fix_sgcmcs.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace LAMMPS_NS {

// struct TagFixSemiGrandCanonicalMCSectorPackForwardComm{};
// struct TagFixSemiGrandCanonicalMCSectorUnPackForwardComm{};

template<class DeviceType>
class FixSemiGrandCanonicalMCSectorKokkos : public FixSemiGrandCanonicalMCSector, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixSemiGrandCanonicalMCSectorKokkos(class LAMMPS *, int, char **);
  ~FixSemiGrandCanonicalMCSectorKokkos() override;

  void init() override; 
  void doMC() override;
  void post_force(int vflag) override;
  double computeEnergyChangeEatom(int flipAtom, int oldSpecies, int newSpecies) override;

  void filter_neighbors();

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d d_ilist;

//   int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d&,
//                        int, int *) override;
//   void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
//   int pack_forward_comm(int, int *, double *, int, int *) override;
//   void unpack_forward_comm(int, int, double *) override;
//   int pack_reverse_comm(int, int, double *) override;
//   void unpack_reverse_comm(int, int *, double *) override;

//   KOKKOS_INLINE_FUNCTION
//   void operator()(TagFixSemiGrandCanonicalMCSectorPackForwardComm, const int&) const;

//   KOKKOS_INLINE_FUNCTION
//   void operator()(TagFixSemiGrandCanonicalMCSectorUnpackForwardComm, const int&) const;

protected:

 DAT::tdual_int_1d k_numneigh_short;
 typename AT::t_int_1d d_numneigh_short;
 HAT::t_int_1d h_numneigh_short;

 DAT::tdual_int_2d k_neighbors_short;
 typename AT::t_int_2d d_neighbors_short;
 HAT::t_int_2d h_neighbors_short;

 DAT::tdual_int_1d k_ilist_short;
 typename AT::t_int_1d d_ilist_short;
 HAT::t_int_1d h_ilist_short;

 int nmax, maxj, cutoff;

 typename AT::t_x_array x;
 typename AT::t_int_1d type;

};

}

#endif
#endif