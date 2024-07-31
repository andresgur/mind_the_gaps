# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   smangham
# @Last modified time: 09-01-2024
import celerite
import warnings
import numpy as np

from mind_the_gaps.models.celerite_models import Lorentzian, DampedRandomWalk, Cosinus, BendingPowerlaw

two_pi = 2 * np.pi
days_to_seconds = 24 * 3600

class ConfigFileError(Exception):
    """Custom exception for configuration file errors."""
    def __init__(self, message):
        super().__init__(message)


def read_config_file(config_file, walkers=32):
    """Read config file with model and parameter initial values and bounds.

    Parameters
    ----------
    config_file:str,
        The config file

    Returns the kernel,
    """
    try:
        model_info = np.genfromtxt(config_file, names=True, dtype="U25,U25,U25,U25",
                                   delimiter="\t", deletechars="")
        if len(np.atleast_1d(model_info))==1:
            model_info = np.array([model_info])


        kernels = np.empty(len(model_info), dtype=celerite.terms.Term)

        outmodels = []
        initial_params = None
        columns = []
        labels = []
        # either nothing if only one kernel or start numbering the terms if more than one
        kernel_string = "" if len(model_info)==1 else "terms[0]:"

        for kernel_counter, row in enumerate(model_info):
            print("Adding %s component" % row["model"])
            outmodels.append("%s" % row["model"])

            w = np.log(two_pi / (np.array(row["P"].split(":")).astype(float) * days_to_seconds))
            S_0 = np.array(row["logS_0"].split(":")).astype(float) #already in log

            if row["model"] == "Lorentzian" or row["model"] =="SHO" or row["model"]=="Powerlaw":
                Q = np.log(np.array(row["Q"].split(":")).astype(float))
                bounds = dict(log_S0=(S_0[0], S_0[2]), log_Q=(Q[0], Q[2]), log_omega0=(w[2], w[0]))

                # create the variables if it's the first kernel
                kernel_columns = ["kernel:%slog_omega0" % kernel_string, "kernel:%slog_S0" %kernel_string, "kernel:%slog_Q"%kernel_string]
                kernel_labels = [r"$P$ (days)", r"log $S_0$", r"log Q"]
                init_model_pars = np.array([np.random.uniform(S_0[0], S_0[2], walkers),
                                         np.random.uniform(Q[0], Q[2], walkers),
                                        np.random.uniform(w[2], w[0], walkers)])
                if initial_params is None:
                    initial_params = init_model_pars
                else:
                    initial_params = np.append(initial_params, init_model_pars, axis=0)

                if row["model"]=="SHO":
                    kernel = celerite.terms.SHOTerm(log_S0=S_0[1], log_Q=Q[1], log_omega0=w[1], bounds=bounds)
                elif row["model"]=="Lorentzian":
                    kernel = Lorentzian(log_S0=S_0[1], log_omega0=w[1], log_Q=Q[1], bounds=bounds)
                elif row["model"]=="Powerlaw":
                    kernel = BendingPowerlaw(log_S0=S_0[1], log_omega0=w[1], log_Q=Q[1], bounds=bounds)

            # two param model components (S_0 and omega)
            elif row["model"] == "DampedRandomWalk" or row["model"] =="Granulation" or row["model"]=="Cosinus":
                bounds = dict(log_S0=(S_0[0], S_0[2]), log_omega0=(w[2], w[0]))

                if row["model"] == "DampedRandomWalk":
                    kernel = DampedRandomWalk(log_S0=S_0[1], log_omega0=w[1],
                                                     bounds=bounds)
                elif row["model"] =="Granulation":
                    Q = 1 / np.sqrt(2)
                    kernel = celerite.terms.SHOTerm(log_S0=S_0[1], log_Q=np.log(Q), log_omega0=w[1],
                                                     bounds=bounds)
                    kernel.freeze_parameter("log_Q")
                elif row["model"]=="Cosinus":
                    kernel = Cosinus(log_S0=S_0[1], log_omega0=w[1], bounds=bounds)

                kernel_labels = [r"log $S_N$", r"log $\omega_N$"]
                kernel_columns = ["kernel:%s%s" %(kernel_string, name) for name in kernel.get_parameter_names()]

                init_model_pars = np.array([np.random.uniform(S_0[0], S_0[2], walkers),
                                            np.random.uniform(w[2], w[0], walkers)])
                if initial_params is None:
                    initial_params = init_model_pars
                else:
                    initial_params = np.append(initial_params, init_model_pars, axis=0)

            elif row["model"]=="Matern32":
                log_rho = np.log(np.array(row["P"].split(":")).astype(float) * days_to_seconds)
                bounds = dict(log_sigma=(S_0[0], S_0[2]), log_rho=(log_rho[0], log_rho[2]))
                kernel = celerite.terms.Matern32Term(log_sigma=S_0[1], log_rho=log_rho[1], eps=1e-7,
                                                 bounds=bounds)

                kernel_columns = ["kernel:%s%s" %(kernel_string, name) for name in kernel.get_parameter_names()]
                kernel_labels = [r"log $\sigma$", r"log $\rho$"]

                init_model_pars = np.array([np.random.uniform(S_0[0], S_0[2], walkers),
                                            np.random.uniform(log_rho[0], log_rho[2], walkers)])
                if initial_params is None:
                    initial_params = init_model_pars

                else:
                    initial_params = np.append(initial_params, init_model_pars, axis=0)

            elif row["model"]=="Jitter":
                bounds = dict(log_sigma=(S_0[0], S_0[2]))
                kernel = celerite.terms.JitterTerm(log_sigma=S_0[1], bounds=bounds)
                kernel_columns = ["kernel:%s%s" %(kernel_string, name) for name in kernel.get_parameter_names()]
                kernel_labels = [r"log $\sigma$"]

                init_model_pars = np.array([np.random.uniform(S_0[0], S_0[2], walkers)])
                if initial_params is None:
                    initial_params = init_model_pars
                else:
                    initial_params = np.append(initial_params, init_model_pars, axis=0)

            else:
                warnings.warn("Component %s unrecognised. Skipping..." % row["model"])

            columns.extend(kernel_columns)
            labels.extend(kernel_labels)
            kernels[kernel_counter] = kernel
            kernel_string = "terms[%d]:" % (kernel_counter + 1)

        total_kernel = np.sum(kernels)
        return total_kernel, initial_params.T, labels, columns, outmodels

    except Exception as e:
        raise ConfigFileError(f"Error reading config file {config_file}: {e}")
