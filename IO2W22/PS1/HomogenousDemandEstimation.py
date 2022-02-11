import numpy as np
import pyblp as blp
import torch
from torch.autograd import Variable
import torch.optim as optim

blp.options.digits = 2
blp.options.verbose = False
nax = np.newaxis


class HomDemEst:

    def __init__(self, data_dict):

        """
            Assumes that data is given with index (Market/Time, Product) and
            log share ratios are computed.
        """

        # Extract dataset and relevant columns from input dictionary.
        data_table = data_dict['Data']
        choice_col = data_dict['Choice Column']
        market_col = data_dict['Market Column']
        share_col = data_dict['Log Share Ratio Column']
        endog_col = data_dict['Endogenous Columns']
        exog_col = data_dict['Exogenous Columns']
        instr_col = data_dict['Instrument Columns']
        add_const = data_dict['Add Constant']

        self.lns = torch.tensor(np.array(data_table[share_col]), dtype=torch.double)
        self.p = torch.tensor(np.array(data_table[endog_col]), dtype=torch.double)

        if self.p.dim() == 1:

            self.p = self.p[:, None]

        self.x = torch.tensor(np.array(data_table[exog_col]), dtype=torch.double)

        if self.x.dim() == 1:

            self.x = self.x[:, None]

        self.z = torch.tensor(np.array(data_table[instr_col]), dtype=torch.double)

        if self.z.dim() == 1:

            self.z = self.z[:, None]

        if add_const:

            self.px = torch.cat((torch.ones(self.p.size()), self.x, self.p), 1)
            self.xz = torch.cat((torch.ones(self.p.size()), self.x, self.z), 1)

        else:

            self.px = torch.cat((self.x, self.p), 1)
            self.xz = torch.cat((self.x, self.z), 1)

        # Extract relevant dimensions.
        self.num_prod = data_table[choice_col].nunique()
        self.num_T = data_table[market_col].nunique()
        self.num_z = self.xz.shape[1]
        self.num_param = self.px.shape[1]

        # Initial GMM weight matrix.
        self.weight_matrix = torch.eye(self.num_z, dtype=torch.double)

    def gmm_eqns(self, coeff):

        """
            Compile linear equations relating log-share ratios to
            endogenous and exogenous co-variates, backing out unobserved
            market-product demand shocks.
        """

        cond = (self.lns - self.px @ coeff)[:, None] * self.xz

        return cond.reshape((self.num_T, self.num_prod, self.num_z))

    def gmm_moments(self, coeff):

        """
            Compile moments for each product and instrument combination.
        """

        moments = self.gmm_eqns(coeff).mean(axis=(0, 1))
        return moments

    def gmm_loss(self, coeff):

        """
            Construct the GMM loss function using the moments corresponding
            to a homogenous market demand system.
        """

        moments = self.gmm_moments(coeff)
        loss = moments[None, :] @ self.weight_matrix @ moments[:, None]
        return loss.squeeze()

    def gmm_jacobian(self, coeff):

        """
           Derive Jacobian matrix of the moments used in the loss function
           with respect to the underlying parameters.
        """

        g = torch.autograd.functional.jacobian(self.gmm_moments,
                                               coeff)
        
        return g

    def return_var_g(self, coeff):

        """
           Derive variance of the equations used in the loss function (the second moments)
        """

        g = self.gmm_eqns(coeff).reshape((self.num_T * self.num_prod, self.num_z))
        return 1/(self.num_T * self.num_prod) * (g.T @ g)

    def one_step_gmm(self) -> torch.tensor:

        β = Variable(torch.zeros(self.num_param, dtype=torch.double), requires_grad=True)
        opt_gmm = optim.Adam([β], lr=0.05)

        # Optimizing over the GMM loss function
        for epoch in range(10000):

            opt_gmm.zero_grad()       # Reset gradient inside the optimizer

            # Compute the objective at the current parameter values.
            loss_round1 = self.gmm_loss(coeff=β)
            loss_round1.backward()    # Gradient computed.
            opt_gmm.step()            # Update parameter values using gradient descent.

        return β

    def efficient_gmm(self) -> torch.tensor:

        β_init = self.one_step_gmm()
        # Construct optimal weight matrix from Step 1.
        self.weight_matrix = torch.inverse(self.return_var_g(β_init)).detach()

        β = Variable(torch.zeros(self.num_param, dtype=torch.double), requires_grad=True)
        # Create new optimization object for Step 2 of GMM.
        opt_gmm2 = optim.Adam([β], lr=0.05)

        # Optimizing over the GMM loss function
        for epoch in range(10000):

            opt_gmm2.zero_grad() # Reset gradient inside the optimizer

            # Compute the objective at the current parameter values.
            loss_round2 = self.gmm_loss(β)
            loss_round2.backward()    # Gradient computed.
            opt_gmm2.step()           # Update parameter values using gradient descent.

        return β

    def run_gmm(self, gmm_type='efficient') -> dict:

        """
        Runs either the one-step or the efficient GMM optimization and returns standard errors.
        """

        if gmm_type == 'one-step':

            β = self.one_step_gmm()
            G = self.gmm_jacobian(β)
            V = self.return_var_g(β)
            W = self.weight_matrix

            bread = torch.inverse(G.T @ W @ G)
            fill = G.T @ W @ V @ W @ G

            return {'Coefficients': β.detach(),
                    'Covariance Matrix': (bread @ fill @ bread).detach()/(self.num_T * self.num_prod)
                    }

        else:

            β = self.efficient_gmm()
            G = self.gmm_jacobian(β)
            V = self.return_var_g(β)

            return {'Coefficients': β.detach(),
                    'Covariance Matrix':
                        torch.inverse(G.T @ torch.inverse(V) @ G).detach()/(self.num_T * self.num_prod)
                    }
