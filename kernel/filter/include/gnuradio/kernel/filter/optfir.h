//
//  Copyright 2004,2005,2009 Free Software Foundation, Inc.
//
//  This file is part of GNU Radio
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//
//

#include <gnuradio/kernel/filter/pm_remez.h>
#include <algorithm>
#include <tuple>
#include <vector>

// Routines for designing optimal FIR filters.

// For a great intro to how all this stuff works, see section 6.6 of
// "Digital Signal Processing: A Practical Approach", Emmanuael C. Ifeachor
// and Barrie W. Jervis, Adison-Wesley, 1993.  ISBN 0-201-54413-X.

namespace gr {
namespace kernel {
namespace filter {

class optfir
{
public:
    /**
     * @brief FIR lowpass filter length estimator.
     *
         FIR lowpass filter length estimator.  freq1 and freq2 are
        normalized to the sampling frequency.  delta_p is the passband
        deviation (ripple), delta_s is the stopband deviation (ripple).

        Note, this works for high pass filters too (freq1 > freq2), but
        doesn't work well if the transition is near f == 0 or f == fs/2

        From Herrmann et al (1973), Practical design rules for optimum
        finite impulse response filters.  Bell System Technical J., 52, 769-99
     *
     * @param freq1
     * @param freq2
     * @param delta_p
     * @param delta_s
     * @return size_t
     */
    static size_t lporder(double freq1, double freq2, double delta_p, double delta_s)
    {
        auto df = abs(freq2 - freq1);
        auto ddp = log10(delta_p);
        auto dds = log10(delta_s);

        double a1 = 5.309e-3;
        double a2 = 7.114e-2;
        double a3 = -4.761e-1;
        double a4 = -2.66e-3;
        double a5 = -5.941e-1;
        double a6 = -4.278e-1;

        double b1 = 11.01217;
        double b2 = 0.5124401;

        auto t1 = a1 * ddp * ddp;
        auto t2 = a2 * ddp;
        auto t3 = a4 * ddp * ddp;
        auto t4 = a5 * ddp;

        auto dinf = ((t1 + t2 + a3) * dds) + (t3 + t4 + a6);
        auto ff = b1 + b2 * (ddp - dds);
        auto n = (size_t)(dinf / df - ff * df + 1);
        return n;
    }

    /**
     * @brief FIR order estimator (lowpass, highpass, bandpass, mulitiband).
     *
        (n, fo, ao, w) = remezord (f, a, dev)
        (n, fo, ao, w) = remezord (f, a, dev, fs)

        (n, fo, ao, w) = remezord (f, a, dev) finds the approximate order,
        normalized frequency band edges, frequency band amplitudes, and
        weights that meet input specifications f, a, and dev, to use with
        the remez command.

        * f is a sequence of frequency band edges (between 0 and Fs/2, where
          Fs is the sampling frequency), and a is a sequence specifying the
          desired amplitude on the bands defined by f. The length of f is
          twice the length of a, minus 2. The desired function is
          piecewise constant.

        * dev is a sequence the same size as a that specifies the maximum
          allowable deviation or ripples between the frequency response
          and the desired amplitude of the output filter, for each band.

        Use remez with the resulting order n, frequency sequence fo,
        amplitude response sequence ao, and weights w to design the filter b
        which approximately meets the specifications given by remezord
        input parameters f, a, and dev:

        b = remez (n, fo, ao, w)

        (n, fo, ao, w) = remezord (f, a, dev, Fs) specifies a sampling frequency Fs.

        Fs defaults to 2 Hz, implying a Nyquist frequency of 1 Hz. You can
        therefore specify band edges scaled to a particular applications
        sampling frequency.

        In some cases remezord underestimates the order n. If the filter
        does not meet the specifications, try a higher order such as n+1
        or n+2.
     *
     * @return std::tuple<size_t, double, double, double>
     */
    static std::
        tuple<size_t, std::vector<double>, std::vector<double>, std::vector<double>>
        remezord(std::vector<double> fcuts,
                 std::vector<double> mags,
                 std::vector<double> devs,
                 double fsamp = 2)
    {
        for (auto& f : fcuts) {
            f = float(f) / fsamp;
        }

        auto nf = fcuts.size();
        auto nm = mags.size();
        auto nd = devs.size();
        auto nbands = nm;

        if (nm != nd)
            throw std::range_error("Length of mags and devs must be equal");

        if (nf != 2 * (nbands - 1)) {
            throw std::range_error("Length of f must be 2 * len (mags) - 2");
        }

        for (size_t i = 0; i < mags.size(); i++) {
            if (mags[i] != 0) { // if not stopband, get relative deviation
                devs[i] = devs[i] / mags[i];
            }
        }

        // separate the passband and stopband edges
        std::vector<double> f1, f2;
        for (size_t i = 0; i < fcuts.size(); i += 2) {
            f1.push_back(fcuts[i]);
        }
        for (size_t i = 1; i < fcuts.size(); i += 2) {
            f2.push_back(fcuts[i]);
        }

        size_t n = 0;
        double min_delta = 2;
        for (size_t i = 0; i < f1.size(); i++) {
            if (f2[i] - f1[i] < min_delta) {
                n = i;
                min_delta = f2[i] - f1[i];
            }
        }

        size_t l = 0;
        if (nbands == 2) {
            // lowpass or highpass case (use formula)
            l = lporder(f1[n], f2[n], devs[0], devs[1]);
        }
        else {
            // bandpass or multipass case
            // try different lowpasses and take the worst one that
            //  goes through the BP specs
            l = 0;
            for (size_t i = 1; i < nbands - 1; i++) {
                size_t l1 = lporder(f1[i - 1], f2[i - 1], devs[i], devs[i - 1]);
                size_t l2 = lporder(f1[i], f2[i], devs[i], devs[i + 1]);
                l = std::max(l, std::max(l1, l2));
            }
        }

        n = (int)ceil(l) - 1; // need order, not length for remez

        // cook up remez compatible result
        std::vector<double> ff = fcuts;
        ff.push_back(1.0);
        ff.insert(ff.begin(), 0.0);
        for (size_t i = 1; i < ff.size() - 1; i++) {
            ff[i] *= 2;
        }


        std::vector<double> aa;
        for (auto& a : mags) {
            aa.push_back(a);
            aa.push_back(a);
            // aa = aa + [a, a]
        }

        auto max_dev = *std::max_element(devs.begin(), devs.end());
        // wts = [1] * len(devs)
        std::vector<double> wts(devs.size());
        std::fill_n(wts.begin(), wts.size(), 1.0);

        for (size_t i = 0; i < wts.size(); i++) {
            wts[i] = max_dev / devs[i];
        }

        return std::make_tuple(n, ff, aa, wts);
    }

    /**
     * @brief Convert a stopband attenuation in dB to an absolute value
     *
     * @return double
     */
    static double stopband_atten_to_dev(double atten_db)
    {
        return pow(10, (-atten_db / 20));
    }

    /**
     * @brief Convert passband ripple spec expressed in dB to an absolute value
     *
     * @return double
     */
    static double passband_ripple_to_dev(double ripple_db)
    {
        return ((pow(10, (ripple_db / 20)) - 1) / (pow(10, (ripple_db / 20)) + 1));
    }

    /**
     * @brief     Builds a low pass filter.
        Args:
            gain: Filter gain in the passband (linear)
            Fs: Sampling rate (sps)
            freq1: End of pass band (in Hz)
            freq2: Start of stop band (in Hz)
            passband_ripple_db: Pass band ripple in dB (should be small, < 1)
            stopband_atten_db: Stop band attenuation in dB (should be large, >= 60)
            nextra_taps: Extra taps to use in the filter (default=2)
     */

    static std::vector<double> low_pass(double gain,
                                        double Fs,
                                        double freq1,
                                        double freq2,
                                        double passband_ripple_db,
                                        double stopband_atten_db,
                                        size_t nextra_taps = 2)
    {
        auto passband_dev = passband_ripple_to_dev(passband_ripple_db);
        auto stopband_dev = stopband_atten_to_dev(stopband_atten_db);
        std::vector<double> desired_ampls{ gain, 0.0 };
        // (n, fo, ao, w)
        auto tup =
            remezord({ freq1, freq2 }, desired_ampls, { passband_dev, stopband_dev }, Fs);
        // The remezord typically under-estimates the filter order, so add 2 taps by
        // default
        auto taps = pm_remez(std::get<0>(tup) + nextra_taps,
                             std::get<1>(tup),
                             std::get<2>(tup),
                             std::get<3>(tup),
                             "bandpass");
        return taps;
    }
    // FIXME - rest of optfir.py
};

} // namespace filter
} // namespace kernel
} // namespace gr