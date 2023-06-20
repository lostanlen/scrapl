from ..core.timefrequency_scattering import joint_timefrequency_scattering_singlepath
from kymatio.scattering1d.core.timefrequency_scattering import jtfs_average_and_format
from kymatio.scattering1d.frontend.base_frontend import TimeFrequencyScatteringBase


class TimeFrequencyScraplBase(TimeFrequencyScatteringBase):
    def scattering_singlepath(self, x, n2, n_fr):
        TimeFrequencyScatteringBase._check_runtime_args(self)
        TimeFrequencyScatteringBase._check_input(self, x)

        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape)
        U_0 = self.backend.pad(
            x, pad_left=self.pad_left, pad_right=self.pad_right)

        filters = [self.phi_f, self.psi1_f, self.psi2_f]
        path = joint_timefrequency_scattering_singlepath(U_0, self.backend,
            filters, self.log2_stride, (self.average=='local'),
            self.filters_fr, self.log2_stride_fr, (self.average_fr=='local'),
            n2, n_fr)

        # Unpad.
        res = max(path['j'][-1], 0)
        path['coef'] = self.backend.unpad(
            path['coef'], self.ind_start[res], self.ind_end[res])

        # Reshape path to batch shape.
        path['coef'] = self.backend.reshape_output(path['coef'],
            batch_shape, n_kept_dims=(1 + (self.format=='joint')))

        return path


__all__ = ['TimeFrequencyScraplBase']
