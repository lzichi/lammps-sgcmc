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
FixStyle(sgcmcs/kk,FixSemiGrandCanonicalMCSector<LMPDeviceType>);
FixStyle(sgcmcs/kk/device,FixSemiGrandCanonicalMCSector<LMPDeviceType>);
FixStyle(sgcmcs/kk/host,FixSemiGrandCanonicalMCSector<LMPHostType>);
// clang-format on
#else

#ifndef FIX_SGCMC_KOKKOS_H
#define FIX_SGCMC_KOKKOS_H

#include "fix_sgcmcs.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixSemiGrandCanonicalMCSectorPackForwardComm{};
struct TagFixSemiGrandCanonicalMCSectorUnPackForwardComm{};
// struct TagFixSemiGrandCanonicalMCSector{};

// TODO: do i need virtual in the baseclass??
template<class DeviceType>
class FixSemiGrandCanonicalMCSectorKokkos : public FixSemiGrandCanonicalMCSector {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  NeighListKokkos<DeviceType> *k_neighborlist;

  FixSemiGrandCanonicalMCSectorKokkos(class LAMMPS *, int, char **);
  ~FixSemiGrandCanonicalMCSectorKokkos() override;

  void init() override;
  void init_list(int id, class NeighList *ptr) override;
  void setup(int) override;
  double compute_vector(int index) override;
  double computeEnergyChangeEatom(int flipAtom, int oldSpecies, int newSpecies) override;

  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d&,
                       int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void doMC() override;

  /* Sectoring method routines */
  void sectoring() override;
  int coords2sector(double *) override;
  void setup_pre_neighbor() override;
  void pre_neighbor() override;


  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixSemiGrandCanonicalMCSectorPackForwardComm, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixSemiGrandCanonicalMCSectorUnPackForwardComm, const int&) const;

//   KOKKOS_INLINE_FUNCTION
//   void operator()(TagFixSemiGrandCanonicalMCSector, const int&) const;

 protected:
  DAT::tdual_int_1d k_speciesCounts;     // dual view
  typename AT::t_int_1d d_speciesCounts; // device view
  HAT::t_int_1d h_speciesCounts;         // host view

  // random number generator in sync with all the processors
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
  
  // random number generator for each processor
  RandPoolWrap rand_pool_local;

  typename AT::t_x_array x;
  typename AT::t_f_array f;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;

  // Used for comm
  int first; 
  typename AT::t_int_1d d_sendlist;
  typename AT::t_xfloat_1d_um v_buf;

  // Neighbor list views
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d d_ilist;
  typename AT::t_int_1d d_numneigh;

  int inum, nlocal, ntypes;

  /* Sectoring method member variables */
  // TODO: don't want stack_foot or forward_stacks to be member variables
  // for now this is b/c operator only accesses member variables
  typename AT::t_int_1d d_rsec;
  typename AT::t_int_1d d_num_atoms_per_sector;
  typename AT::t_int_1d d_atoms_in_sector;
//   typename AT::t_int_1d d_stack_foot;
//   typename AT::t_int_1d d_forward_stacks;

};

}

#endif
#endif
