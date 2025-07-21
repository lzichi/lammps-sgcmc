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

#include "fix_sgcmc_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
FixSemiGrandCanonicalMCKokkos<DeviceType>::FixSemiGrandCanonicalMCKokkos(LAMMPS *lmp, int narg, char **arg) :
 FixSemiGrandCanonicalMC(lmp, narg, arg), rand_pool(seed), rand_pool_local(seed + comm->me)
{
    kokkosable = 1;
    atomKK = (AtomKokkos *) atom;
    execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

}

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
FixSemiGrandCanonicalMCKokkos<DeviceType>::~FixSemiGrandCanonicalMCKokkos()
{
    if (copymode) return;
}

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
void FixSemiGrandCanonicalMCKokkos<DeviceType>::init()
{
    FixSemiGrandCanonicalMC::init();
    Kokkos::resize(k_speciesCounts, atomKK->ntypes + 1);
}

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
void FixSemiGrandCanonicalMCKokkos<DeviceType>::init_list()
{
    FixSemiGrandCanonicalMC::init_list();
    NeighListKokkos<DeviceType>* k_neighborList = static_cast<NeighListKokkos<DeviceType>*>(neighborList);

    // TODO: where should this actually go
    x = atomKK->k_x.view<DeviceType>();
    f = atomKK->k_f.view<DeviceType>();
    type = atomKK->k_type.view<DeviceType>();

    NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);

    d_numneigh = k_list->d_numneigh;
    d_neighbors = k_list->d_neighbors;
    d_ilist = k_list->d_neighors;
    inum = list->inum;
}

/*********************************************************************
 *********************************************************************/

template<class DeviceType>
void FixSemiGrandCanonicalMCKokkos<DeviceType>::post_force(int vflag) 
{
    if ((update->ntimestep % nevery_mdsteps) == 0)
        doMC();

}

/*********************************************************************
 * Lets the fix report one of its internal state variables to LAMMPS.
 *********************************************************************/
 double FixSemiGrandCanonicalMCKokkos<DeviceType>::compute_vector(int index)
 {
    if (index == 0) return nAcceptedSwaps;
    if (index == 1) return nRejectedSwaps;
    index -=1;
    int totalAtoms = 0;
    
    // TODO: is copymode necessary ????
    // TODO: should i use k_speciesCouns
    // TODO: should i use atomKK
    // TODO: can you do k_speciesCounts.size_t
    // TODO: how do you cast views
    copymode = 1;
    Kokkos::parallel_reduce("compute_vector:totalAtoms", k_speciesCounts.size_t, 
                        KOKKOS_LAMBDA (const int& i, int& ltotalAtoms) 
                        {
                            ltotalAtoms += k_speciesCounts[i];
                        }, totalAtoms);
    copymode = 0;

    if (index <= atomKK->ntypes)
        return (double)k_speciesCounts(index) / (totalAtoms > 0 ? totalAtoms : 1);

    
    return 0.0;
 }

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixSemiGrandCanonicalMCKokkos::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                            DAT::tdual_xfloat_1d &buf,
                                                            int /*pbc_flag*/, int * /*pbc*/)
{
    d_sendlist = k_sendlist.view<DeviceType>();
    v_buf = buf.view<DeviceType>();
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixSemiGrandCanonicalMCPackForwardComm>(0, n), *this);
    return n;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSemiGrandCanonicalMCKokkos<DeviceType>::operator()(TagFixSemiGrandCanonicalMCPackForwardComm, const int &i) const {
    // TODO: is this the correct type to use?
    int j = d_sendlist(i);
    v_buf[i] =  type[j];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCKokkos::unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_xfloat_1d &buf)
{
    first = first_in;
    v_buf = buf.view<DeviceType>();
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixSemiGrandCanonicalMCUnpackForwardComm>(0,n), *this);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSemiGrandCanonicalMCKokkos<DeviceType>::operator()(TagFixSemiGrandCanonicalMCUnpackForwardComm, const int &i) const {
    // TODO: why does original fix go from last to first instead of first to last?
    // TODO: which atom to modify atom->ntype or atomKK->k_type.view<DeviceType>()
    type[i + first] = (int)v_buf[i];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixSemiGrandCanonicalMCKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
                                                                 int /*pbc_flag*/, int * /*pbc*/)
{
    // TODO: do i need to sync anything? i think not becasue not dual
    int i,j;

    for (i = 0; i < n; i++) {
        j = list[i];
        // TODO: is this the correct type to access?
        buf[i] = atom->type[j];
    }
    return n;
}            

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf);
{
    // TODO: do i need to sync or modify host?
    for (int i = 0; i < n; i++) {
        atom->type[i + first] = buf[i];
    }
}
                                 
/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixSemiGrandCanonicalMCKokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
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
void FixSemiGrandCanonicalMCKokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
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

bool FixSemiGrandCanonicalMCKokkos<DeviceType>::placeSamplingWindow()
{
    bool oversizeWindow = false;

    // TODO: how to deal with member variables in a kokkos lambda
    // TODO: how to deal with domain and kokkos
    typename AT::tdual_int_1d d_sublo = domain->sublo.view<DeviceType>;
    typename AT::tdual_int_1d d_subhi = domain->subhi.view<DeviceType>;

    // Align the sampling window to one of the 8 corners of the processor cell.
    const size_t N0 = 3;
    Kokkos::View<double*> k_smaplingWindowLo("k_samplingWindowLo", N0); 
    Kokkos::View<double*> k_samplingWindowHi("k_samplingWindowHi", N0);
    Kokkos::View<double*> k_margin("margin", N0);

    // TODO: decide if this is worth parallelizing
    // if so, then deal with SamplingWindowPosition
    for (int i = 0; i < 3; i++)
    {
        margin(i) = interactionRadius * 2.0;
        if (samplingWindoUserSize > 0.0) {
            margin(i) = (domain->subhi[i] - domain->sublo[i]) * (1.0 - samplingWindoUserSize);
            if(margin(i) < interactionRadius * 2.0)
                oversizeWindow = true;
        }

        // TODO: can you use double or should this be a *_FLOAT thing from Kokkos_types
        double shift = (double)((samplingWindowPosition >> i) & 1) * margin(i);
        samplingWindowLo(i) = d_sublo(i) + shift;
        samplingWindowHi(i) = d_subhi(i) + shift - margin(i);

        if (k_smaplingWindowHi(i) - samplingWindowLo(i) + 1e-6 < (d_subhi(i) - d_sublo(i)) * 0.5) {
            error->one(FLERR, "Per-node simulation cell is too small for fix sgcmc. Processor cell size must be at least 4 times cutoff radius.");
        }
    }

    // Increase counter by one
    SamplingWindowPosition += 1;

    // Compile a list of atoms that are inside the sampling window

    // TODO: find better way to deal with SamplingWindowAtoms
    // TODO: why does it only reserve size nlocal
    // TODO: probably bad to allocate a view each time
    // TODO: should i be accessing the k_list here?
    // Each atom can have up to multiplicity 8
    Kokkos::reserve(k_samplingWindowAtomsTemp, inum * 8);
    Kokkos::parallel_for("FillArray", inum * 8, KOKKOS_LAMBDA(const int i) {
        k_samplingWindowAtomsTemp(i) = -1;
    });

    // Optionally, you can print the values to verify
    Kokkos::fence(); // Ensure all operations are complete

    numSamplingWindowAtoms = 0;
    numFixAtomsLocal = 0;
    
    // TODO: is this the correct type? should this be a member variable?
    typename AT::t_int_1d_randomread d_mask = atom->mask.view<DeviceType>();
    int sampleAtomsCount = 0;

    Kokkos::parallel_for("placeSamplingWindow:listAtomsInWindow", inum, KOKKOS_LAMBDA (const int ii) {
        int i = d_ilist(ii);
        if (d_mask(i) & groupbit) {
            numFixAtomsLocal++;
            
            if (x(i, 0) >= k_smaplingWindowLo(0) && x(i, 0) < k_samplingWindowHi(0) &&
                x(i, 1) >= k_smaplingWindowLo(1) && x(i, 1) < k_samplingWindowHi(1) &&
                x(i, 2) >= k_smaplingWindowLo(2) && x(i, 2) < k_samplingWindowHi(2)) {

                    int multiplicity = 1;
                    for (int k = 0; k < 3; k++) {
                        if (x(i, k) < d_sublo(k) + k_margin(k) ||
                            x(i, k) > d_subhi(k) - k_margin(k))
                            multiplicity *= 2;
                    }
                    sampleAtomsCount += multiplicity;

                    for (int m = 0; m < multiplicity; m++)
                        k_samplingWindowAtomsTemp(ii + m) = ii;

                    numSamplingWindowAtoms++;
            }
        }
    });

    // fill in the gaps
    // TODO: figure out the best way!
    Kokkos::reserve(k_samplingWindowAtoms, numFixAtomsLocal);
    int index = 0;

    // TODO: idk how to parallelize without a race condition

    //Kokkos::parallel_for("placeSamplingWindow:fillAtomsInWindow", inum*8, KOKKOS_LAMBDA (const int ii) {
    for (int ii = 0; ii < inum * 8; ii++) {
        if (k_samplingWindowAtomsTemp[ii] != -1) {
            k_samplingWindowAtoms[index] = k_samplingWindowAtomsTemp[ii];
            index += 1;
        }
    }

    return oversizeWindow;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSemiGrandCanonicalMCKokkos<DeviceType>::doMC() {

    // TODO: decide if mask should be a member variable
    typename AT::t_int_1d_randomread d_mask = atom->mask.view<DeviceType>();

    // Reset counters
    int nAcceptedSwapsLocal = 0;
    int nRejectedSwapsLocal = 0;

    int oldSpecies, newSpecies;

    // Local and global change in number of atoms of each species
    // TODO: is this the correct kokkos type to use??
    // TODO: can i access atom->ntypes?
    typename AT::t_int_1d d_deltaN("d_deltaN", atom->ntypes + 1);
    typename AT::t_int_1d d_deltaNGlobal("d_deltaNGlobal", atom->ntypes + 1);

    for (int i = 0; i < numSamplingWindowsMoves; i++) {
        
        bool oversizeWindow = placeSamplingWindow();

        int nDice = (int)(swap_fraction * numFixAtomsLocal / numSamplingWindowMoves);

        int largestnDice;
        MPI_Allreduce(&nDice, &largestnDice, 1, MPI_INT, MPI_MAX, world);

        // The probability to do one swap step.
        double diceProbability = (double)nDice / (double)largestnDice;

        // parallelize the attempts inside the domain of the node
        // TODO: should i have this range policy and use a functor?
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixSemiGrandCanonicalMC>(0, largestnDice), *this);

        if(!oversizeWindow)
            communicateRhoAndTypes();
    }

    // TODO: should these types be kokkos??
    // MPI sum total number of accepted/rejected swaps.
    MPI_Allreduce(&nAcceptedSwapsLocal, &nAcceptedSwaps, 1, MPI_INT, MPI_SUM, world);
    MPI_ALLreduce(&nRejectedSwapsLocal, &nRejectedSwaps, 1, MPI_INT, MPI_SUM, world);

    if (kappa == 0.0 && !serialMode) {
        // TODO: should these types be kokkos??
        // TODO: can you do an MPI reduce over kokkos views
        // TODO: can you parallelize this without a race condition 
        // TODO: should localSpeciesCounts be kokkos view or dual view maybe
        for( int i = 0; i < nlocal; i++) {
            if(d_mask(i) & groupbit)
                localSpeciesCounts[type[i]]++; // TODO: is this the same thing as what is in fix_sgcmc
        }

        MPI_ALLreduce(localSpeciesCounts.data(), speciesCounts.data(), localSpeciesCounts.size(), MPI_INT, MPI_SUM, world);
    }
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSemiGrandCanonicalMCKokkos<DeviceType>::operator()(TagFixSemiGrandCanonicalMC, const int &i) {
    
    double deltaE = 0;
    std::fill(d_deltaN.begin(), d_deltaN.end(), 0);
    int selectedAtom = -1, selectedAtomNL = -1;
    rand_type rand_gen = rand_pool_local.get_state();

    if(rand_gen.drand() <= diceProbability) {
        int index = (int)(rand_gen.drand() * (double)k_samplingWindowAtoms.extent(0));
        selectedAtomNL = k_samplingWindowAtoms(index);

        selectedAtom = d_ilist(selectedAtomNL);
        oldSpecies = type(selectedAtom);

        // TODO can i access atom->ntypes
        if (atom->ntypes > 2) {
            newSpecies = (int)(rand_gen.drand() * (atom->ntypes - 1)) + 1;
            if (newSpecies >= oldSpecies) newSpecies++;
        } else {
            newSpecies = (oldSpecies == 1) ? 2 : 1;
        }
        // this could be a race condition
        d_deltaN(oldSpecies) = -1;
        d_deltaN(newSpecies) = +1;

        fence();

        // Perform inner MC acceptance test
        
    }
}


