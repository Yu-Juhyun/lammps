/* ----------------------------------------------------------------------
   MACE-QEq Extension for LAMMPS
   
   This file provides QEq charge extraction functionality for MACE models
   that predict atomic charges.
------------------------------------------------------------------------- */

#ifndef LMP_MACE_QEQ_H
#define LMP_MACE_QEQ_H

#include <torch/torch.h>
#include <torch/script.h>

namespace LAMMPS_NS {
namespace MACEQEq {

/**
 * @brief Add total_charge to the input dictionary for QEq models
 * 
 * @param input The input dictionary to modify
 * @param total_charge Total system charge (default: 0.0 for neutral)
 * @param dtype Torch dtype to use
 * @param device Torch device to use
 */
inline void add_total_charge_input(
    c10::Dict<std::string, torch::Tensor>& input,
    double total_charge,
    torch::Dtype dtype,
    c10::Device device)
{
    auto total_charge_tensor = torch::tensor({total_charge}, dtype).to(device);
    input.insert("total_charge", total_charge_tensor);
}

/**
 * @brief Extract charges from model output and update LAMMPS atom charges
 * 
 * @param output Model output dictionary
 * @param q Pointer to LAMMPS atom charge array (atom->q)
 * @param q_flag Whether charges are enabled in LAMMPS (atom->q_flag)
 * @param ilist Neighbor list ilist
 * @param inum Number of local atoms
 * @return true if charges were successfully extracted, false otherwise
 */
inline bool extract_charges(
    const c10::impl::GenericDict& output,
    double* q,
    int q_flag,
    int* ilist,
    int inum)
{
    if (!q_flag || q == nullptr) {
        return false;
    }
    
    if (!output.contains("charges")) {
        return false;
    }
    
    auto charges = output.at("charges").toTensor().cpu();
    
    #pragma omp parallel for
    for (int ii = 0; ii < inum; ++ii) {
        int i = ilist[ii];
        q[i] = charges[i].item<double>();
    }
    
    return true;
}

/**
 * @brief Parse total_charge from pair_style argument string
 * 
 * @param arg Argument string (e.g., "total_charge=1.0")
 * @param total_charge Output variable for parsed charge
 * @return true if argument was a total_charge specification, false otherwise
 */
inline bool parse_total_charge_arg(const char* arg, double& total_charge)
{
    if (strncmp(arg, "total_charge=", 13) == 0) {
        total_charge = atof(arg + 13);
        return true;
    }
    return false;
}

/**
 * @brief Check if model output contains QEq-related fields
 * 
 * @param output Model output dictionary
 * @return true if QEq fields are present
 */
inline bool has_qeq_output(const c10::impl::GenericDict& output)
{
    return output.contains("charges") || 
           output.contains("chi") || 
           output.contains("eta");
}

/**
 * @brief Extract electronegativity (chi) from model output
 * 
 * @param output Model output dictionary
 * @return Tensor of chi values, or empty tensor if not available
 */
inline torch::Tensor extract_chi(const c10::impl::GenericDict& output)
{
    if (output.contains("chi")) {
        return output.at("chi").toTensor().cpu();
    }
    return torch::Tensor();
}

/**
 * @brief Extract hardness (eta) from model output
 * 
 * @param output Model output dictionary
 * @return Tensor of eta values, or empty tensor if not available
 */
inline torch::Tensor extract_eta(const c10::impl::GenericDict& output)
{
    if (output.contains("eta")) {
        return output.at("eta").toTensor().cpu();
    }
    return torch::Tensor();
}

/**
 * @brief Extract chi and eta from model output and store in fix property/atom arrays
 * 
 * @param output Model output dictionary
 * @param chi_ptr Pointer to d_chi array from fix property/atom (can be nullptr)
 * @param eta_ptr Pointer to d_eta array from fix property/atom (can be nullptr)
 * @param ilist Neighbor list ilist
 * @param inum Number of local atoms
 * @return true if at least one property was extracted
 * 
 * Usage in LAMMPS input:
 *   fix chi_prop all property/atom d_chi
 *   fix eta_prop all property/atom d_eta
 *   dump 1 all custom 100 dump.lammpstrj id type x y z q d_chi d_eta
 */
inline bool extract_chi_eta(
    const c10::impl::GenericDict& output,
    double* chi_ptr,
    double* eta_ptr,
    int* ilist,
    int inum)
{
    bool extracted = false;
    
    // Extract chi if available and pointer is valid
    if (chi_ptr != nullptr && output.contains("chi")) {
        auto chi = output.at("chi").toTensor().cpu();
        #pragma omp parallel for
        for (int ii = 0; ii < inum; ++ii) {
            int i = ilist[ii];
            chi_ptr[i] = chi[i].item<double>();
        }
        extracted = true;
    }
    
    // Extract eta if available and pointer is valid
    if (eta_ptr != nullptr && output.contains("eta")) {
        auto eta = output.at("eta").toTensor().cpu();
        #pragma omp parallel for
        for (int ii = 0; ii < inum; ++ii) {
            int i = ilist[ii];
            eta_ptr[i] = eta[i].item<double>();
        }
        extracted = true;
    }
    
    return extracted;
}

/**
 * @brief Extract electrostatic energy from model output
 * 
 * @param output Model output dictionary
 * @return Electrostatic energy value, or 0.0 if not available
 */
inline double extract_electrostatic_energy(const c10::impl::GenericDict& output)
{
    if (output.contains("energy_electrostatic")) {
        return output.at("energy_electrostatic").toTensor().cpu().item<double>();
    }
    return 0.0;
}

}  // namespace MACEQEq
}  // namespace LAMMPS_NS

#endif  // LMP_MACE_QEQ_H
