#!/usr/bin/env luajit
------------------------------------------------------------------------------
-- Sum scalars in parallel using OpenCL.
-- Copyright © 2013–2015 Peter Colberg.
-- Distributed under the MIT license. (See accompanying file LICENSE.)
------------------------------------------------------------------------------

local cl = require("opencl")
local templet = require("templet")
local random = require("random")
local bit = require("bit")
local ffi = require("ffi")

local platform = cl.get_platforms()[1]
local device = platform:get_devices()[1]
local context = cl.create_context({device})
local queue = context:create_command_queue(device)

local N = 1000000
local d_v = context:create_buffer(N * ffi.sizeof("cl_double3"))
local v = ffi.cast("cl_double3 *", queue:enqueue_map_buffer(d_v, true, "write"))
for i = 0, N - 1, 2 do
  v[i].x, v[i + 1].x = random.normal()
  v[i].y, v[i + 1].y = random.normal()
  v[i].z, v[i + 1].z = random.normal()
end
queue:enqueue_unmap_mem_object(d_v, v)

local work_size = 128
local num_groups = 512
local glob_size = num_groups * work_size

local temp = templet.loadstring[[
#ifndef CL_VERSION_1_2
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
__kernel void sum(__global const double3 *restrict d_v,
                  __global double *restrict d_en)
{
  const long gid = get_global_id(0);
  const long lid = get_local_id(0);
  const long wid = get_group_id(0);
  __local double l_en[${work_size}];
  double en = 0;
  for (long i = gid; i < ${N}; i += ${glob_size}) {
    en += dot(d_v[i], d_v[i]);
  }
  l_en[lid] = en;
  barrier(CLK_LOCAL_MEM_FENCE);
  |local i = work_size
  |while i > 1 do
  |  i = i / 2
  if (lid < ${i}) l_en[lid] += l_en[lid + ${i}];
  barrier(CLK_LOCAL_MEM_FENCE);
  |end
  if (lid == 0) d_en[wid] = l_en[0];
}
]]

local source = temp({N = N, work_size = work_size, glob_size = glob_size})
local program = context:create_program_with_source(source)
local status, err = pcall(function() return program:build() end)
io.stderr:write(program:get_build_info(device, "log"))
if not status then error(err) end

local kernel = program:create_kernel("sum")
local d_en = context:create_buffer(num_groups * ffi.sizeof("cl_double"))
kernel:set_arg(0, d_v)
kernel:set_arg(1, d_en)
queue:enqueue_ndrange_kernel(kernel, nil, {glob_size}, {work_size})

local sum = 0
local en = ffi.cast("cl_double *", queue:enqueue_map_buffer(d_en, true, "read"))
for i = 0, num_groups - 1 do sum = sum + en[i] end
queue:enqueue_unmap_mem_object(d_en, en)
print(0.5 * sum / N)