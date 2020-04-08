/*
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "rt/rt_api.h"
#include "stdio.h"
#include "../cl/cluster.h"

/** 
 * \brief Fabric main
 */
int main(void)
{

#ifdef POWER

    //SEQ
    //                             0    1    2    3    4    5    6    7    8    9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24
    unsigned int voltage_list[] = {800, 850, 850, 850, 900, 900, 900, 950, 950, 950, 1000, 1000, 1000, 1000, 1050, 1050, 1050, 1050, 1100, 1100, 1100, 1100, 1100};
    unsigned int freq_list[]    = {50,  50,  100, 150, 50,  100, 150, 50,  100, 150, 50,   100,  150,  200,  50,   100,  150,  200,  50,   100,  150,  200,  250};
    //MAX
    //unsigned int voltage_list[] = {1200};
    //unsigned int freq_list[] = {350};
    unsigned int num_points = sizeof(voltage_list) / sizeof(unsigned int);
    unsigned int freq_factor = 1000000;

    rt_freq_set(RT_FREQ_DOMAIN_FC, 50 * freq_factor);

    while(1) {

        rt_time_wait_us(500000);
        printf("start\n");
        rt_time_wait_us(500000);

        for (unsigned int i = 0; i < num_points; i++) {

            unsigned int voltage = voltage_list[i];
            unsigned int freq = freq_list[i] * freq_factor;

            rt_freq_set(RT_FREQ_DOMAIN_CL, freq);
            rt_voltage_force(RT_VOLTAGE_DOMAIN_MAIN, voltage, NULL);

            // wait until the voltage is set (90ms)
            rt_time_wait_us(90000);

            // mount the cluster, and wait until the cluster is mounted
            rt_cluster_mount(1, 0, 0, NULL);

            // call the cluster entry point, and wait unitl it is finished
            rt_cluster_call(NULL, 0, cluster_entry, NULL, NULL, 0, 0, 0, NULL);

            // unmount the cluster
            rt_cluster_mount(0, 0, 0, NULL);

            // wait for 100ms
            rt_time_wait_us(10000);

        }

    }

#else//POWER

    // change the clock frequency

    // 100MHz
    int freq = 100000000;

    rt_freq_set(RT_FREQ_DOMAIN_FC, freq);
    rt_freq_set(RT_FREQ_DOMAIN_CL, freq);


    printf("fc::main::main\n");

    // mount the cluster, and wait until the cluster is mounted
    rt_cluster_mount(1, 0, 0, NULL);

    // call the cluster entry point, and wait unitl it is finished
    rt_cluster_call(NULL, 0, cluster_entry, NULL, NULL, 0, 0, 0, NULL);

    // unmount the cluster
    rt_cluster_mount(0, 0, 0, NULL);

#endif//POWER


    return 0;
}
