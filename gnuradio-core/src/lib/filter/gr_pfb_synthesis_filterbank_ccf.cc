/* -*- c++ -*- */
/*
 * Copyright 2010 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gr_pfb_synthesis_filterbank_ccf.h>
#include <gri_fft.h>
#include <gr_io_signature.h>
#include <cstdio>
#include <cstring>

gr_pfb_synthesis_filterbank_ccf_sptr gr_make_pfb_synthesis_filterbank_ccf 
    (unsigned int numchans, const std::vector<float> &taps)
{
  return gr_pfb_synthesis_filterbank_ccf_sptr 
    (new gr_pfb_synthesis_filterbank_ccf (numchans, taps));
}


gr_pfb_synthesis_filterbank_ccf::gr_pfb_synthesis_filterbank_ccf
    (unsigned int numchans, const std::vector<float> &taps)
  : gr_sync_interpolator ("pfb_synthesis_filterbank_ccf",
			  gr_make_io_signature (1, numchans, sizeof(gr_complex)),
			  gr_make_io_signature (1, 1, sizeof(gr_complex)),
			  numchans),
    d_updated (false), d_numchans(numchans), d_state(0)
{
  d_filters = std::vector<gri_fir_filter_with_buffer_ccf*>(2*d_numchans);

  // Create an FIR filter for each channel and zero out the taps
  std::vector<float> vtaps(0, 2*d_numchans);
  for(unsigned int i = 0; i < 2*d_numchans; i++) {
    d_filters[i] = new gri_fir_filter_with_buffer_ccf(vtaps);
  }

  // Now, actually set the filters' taps
  set_taps2(taps);

  // Create the IFFT to handle the input channel rotations
  d_fft = new gri_fft_complex (2*d_numchans, false);
  memset(d_fft->get_inbuf(), 0, 2*d_numchans*sizeof(gr_complex));

  set_output_multiple(d_numchans);
}

gr_pfb_synthesis_filterbank_ccf::~gr_pfb_synthesis_filterbank_ccf ()
{
  for(unsigned int i = 0; i < d_numchans; i++) {
    delete d_filters[i];
  }
}

void
gr_pfb_synthesis_filterbank_ccf::set_taps (const std::vector<float> &taps)
{
  unsigned int i,j;

  unsigned int ntaps = taps.size();
  d_taps_per_filter = (unsigned int)ceil((double)ntaps/(double)d_numchans);

  // Create d_numchan vectors to store each channel's taps
  d_taps.resize(d_numchans);

  // Make a vector of the taps plus fill it out with 0's to fill
  // each polyphase filter with exactly d_taps_per_filter
  std::vector<float> tmp_taps;
  tmp_taps = taps;
  while((float)(tmp_taps.size()) < d_numchans*d_taps_per_filter) {
    tmp_taps.push_back(0.0);
  }
 
  // Partition the filter
  for(i = 0; i < d_numchans; i++) {
    // Each channel uses all d_taps_per_filter with 0's if not enough taps to fill out
    d_taps[i] = std::vector<float>(d_taps_per_filter, 0);
    for(j = 0; j < d_taps_per_filter; j++) {
      d_taps[i][j] = tmp_taps[i + j*d_numchans];  // add taps to channels in reverse order
    }
    
    // Build a filter for each channel and add it's taps to it
    d_filters[i]->set_taps(d_taps[i]);
  }

  // Set the history to ensure enough input items for each filter
  set_history (d_taps_per_filter+1);

  d_updated = true;
}

void
gr_pfb_synthesis_filterbank_ccf::set_taps2 (const std::vector<float> &taps)
{
  unsigned int i,j;
  int state = 0;

  unsigned int ntaps = taps.size();
  d_taps_per_filter = (unsigned int)ceil((double)ntaps/(double)d_numchans);

  // Create d_numchan vectors to store each channel's taps
  d_taps.resize(2*d_numchans);

  // Make a vector of the taps plus fill it out with 0's to fill
  // each polyphase filter with exactly d_taps_per_filter
  std::vector<float> tmp_taps;
  tmp_taps = taps;
  while((float)(tmp_taps.size()) < d_numchans*d_taps_per_filter) {
    tmp_taps.push_back(0.0);
  }
 
  // Partition the filter
  for(i = 0; i < d_numchans; i++) {
    // Each channel uses all d_taps_per_filter with 0's if not enough taps to fill out
    d_taps[i] = std::vector<float>(d_taps_per_filter, 0);
    d_taps[d_numchans+i] = std::vector<float>(d_taps_per_filter, 0);
    state = 0;
    for(j = 0; j < d_taps_per_filter; j++) {
      // add taps to channels in reverse order
      // Zero out every other tap
      if(state == 0) {
	d_taps[i][j] = tmp_taps[i + j*d_numchans];  
	d_taps[d_numchans + i][j] = 0;
	state = 1;
      }
      else {
	d_taps[i][j] = 0;
	d_taps[d_numchans + i][j] = tmp_taps[i + j*d_numchans];
	state = 0;
      }
    }
    
    // Build a filter for each channel and add it's taps to it
    d_filters[i]->set_taps(d_taps[i]);
    d_filters[d_numchans + i]->set_taps(d_taps[d_numchans + i]);
  }

  // Set the history to ensure enough input items for each filter
  set_history (d_taps_per_filter+1);

  d_updated = true;
}

void
gr_pfb_synthesis_filterbank_ccf::print_taps()
{
  unsigned int i, j;
  for(i = 0; i < 2*d_numchans; i++) {
    printf("filter[%d]: [", i);
    for(j = 0; j < d_taps_per_filter; j++) {
      printf(" %.4e", d_taps[i][j]);
    }
    printf("]\n\n");
  }
}


int
gr_pfb_synthesis_filterbank_ccf::work (int noutput_items,
				       gr_vector_const_void_star &input_items,
				       gr_vector_void_star &output_items)
{
  gr_complex *in = (gr_complex*) input_items[0];
  gr_complex *out = (gr_complex *) output_items[0];

  if (d_updated) {
    d_updated = false;
    return 0;		     // history requirements may have changed.
  }

  unsigned int n, i;
  for(n = 0; n < noutput_items/d_numchans; n++) {
    for(i = 0; i < d_numchans; i++) {
      in = (gr_complex*)input_items[i];
      d_fft->get_inbuf()[i] = in[n];
    }
    
    // spin through IFFT
    d_fft->execute();

    // Output is sum of two filters, but the input buffer to the filters must be circularly
    // shifted by numchans every time through, done by using d_state to determine which IFFT
    // buffer position to pull from.
    for(i = 0; i < d_numchans; i++) {
      out[i]  = d_filters[i]->filter(d_fft->get_outbuf()[d_state*d_numchans+i]);
      out[i] += d_filters[d_numchans+i]->filter(d_fft->get_outbuf()[(d_state^1)*d_numchans+i]);
    }
    d_state ^= 1;
    
    out += d_numchans;
  }

  return noutput_items;
}
