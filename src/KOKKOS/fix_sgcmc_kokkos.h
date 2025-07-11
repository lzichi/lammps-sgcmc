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
FixStyle(sgcmc/kk,FixSemiGrandCanonicalMC<LMPDeviceType>);
FixStyle(sgcmc/kk/device,FixSemiGrandCanonicalMC<LMPDeviceType>);
FixStyle(sgcmc/kk/host,FixSemiGrandCanonicalMC<LMPHostType>);
// clang-format on
#else

#ifndef FIX_SGCMC_KOKKOS_H
#define FIX_SGCMC_KOKKOS_H

#include "fix_sgcmc.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixSemiGrandCanonicalMCPackForwardComm{};
struct TagFixSemiGrandCanonicalMCUnPackForwardComm{};

// TODO: define structs
template<class DeviceType>
class FixSemiGrandCanonicalMCKokkos : public FixSemiGrandCanonicalMC {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixSemiGrandCanonicalMCKokkos(class LAMMPS *, int, char **);
  ~FixSemiGrandCanonicalMCKokkos() override;
  void init() override;
  void init_list()
  void post_force(int vflag) override;
  double compute_vector(int index) override;

  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d&,
                       int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  bool placeSamplingWindow();

  void doMC();


 private:
  // AT templated and defined when template is instantiated
  // Dual View of ints, 1D, see kokkos_type.h for definition
  typename AT::tdual_int_1d k_speciesCounts; 

  // TODO: is this the correct  type? should this be a member variable?
  // this probably does not need to be a dual view? idk
  typename AT::tdual_int_1d k_samplingWindowAtoms;

  // TODO: this should be inherited, so i probably dont need this?
  // TODO: sam for numFixAtomsLocal
  int numSamplingWindowAtoms;



 protected:
  typename AT::t_x_array x;
  typename AT::t_f_array f;
  typename AT::t_int_1d type;

  // Used for comm
  int first; 
  typename AT::t_int_1d d_sendlist;
  typename AT::t_xfloat_1d_um v_buf;

  // Neighbor list views
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d d_ilist;
  typename AT::t_int_1d d_numneigh;

  int inum, nlocal;


};

}

#endif
#endif
