/**
  * Copyright 2023 Google LLC
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *      http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

module "network1" {
  source          = "./modules/embedded/modules/network/vpc"
  deployment_name = var.deployment_name
  project_id      = var.project_id
  region          = var.region
}

module "debug_node_group" {
  source                 = "./modules/embedded/community/modules/compute/schedmd-slurm-gcp-v5-node-group"
  labels                 = var.labels
  machine_type           = "n2-standard-2"
  node_count_dynamic_max = 4
  project_id             = var.project_id
}

module "debug_partition" {
  source               = "./modules/embedded/community/modules/compute/schedmd-slurm-gcp-v5-partition"
  deployment_name      = var.deployment_name
  enable_placement     = false
  exclusive            = false
  is_default           = true
  node_groups          = flatten([module.debug_node_group.node_groups])
  partition_name       = "debug"
  project_id           = var.project_id
  region               = var.region
  subnetwork_self_link = module.network1.subnetwork_self_link
  zone                 = var.zone
}

module "a21g_node_group" {
  source                 = "./modules/embedded/community/modules/compute/schedmd-slurm-gcp-v5-node-group"
  labels                 = var.labels
  machine_type           = "a2-highgpu-1g"
  node_count_dynamic_max = 20
  project_id             = var.project_id
}

module "a2_partition" {
  source               = "./modules/embedded/community/modules/compute/schedmd-slurm-gcp-v5-partition"
  deployment_name      = var.deployment_name
  node_groups          = flatten([module.a21g_node_group.node_groups])
  partition_name       = "a2"
  project_id           = var.project_id
  region               = var.region
  subnetwork_self_link = module.network1.subnetwork_self_link
  zone                 = var.zone
}

module "v1001g_node_group" {
  source = "./modules/embedded/community/modules/compute/schedmd-slurm-gcp-v5-node-group"
  guest_accelerator = [{
    count = 1
    type  = "nvidia-tesla-v100"
  }]
  labels                 = var.labels
  machine_type           = "n1-standard-8"
  node_count_dynamic_max = 20
  project_id             = var.project_id
}

module "v100_partition" {
  source               = "./modules/embedded/community/modules/compute/schedmd-slurm-gcp-v5-partition"
  deployment_name      = var.deployment_name
  node_groups          = flatten([module.v1001g_node_group.node_groups])
  partition_name       = "v100"
  project_id           = var.project_id
  region               = var.region
  subnetwork_self_link = module.network1.subnetwork_self_link
  zone                 = var.zone
}

module "slurm_controller" {
  source                        = "./modules/embedded/community/modules/scheduler/schedmd-slurm-gcp-v5-controller"
  deployment_name               = var.deployment_name
  disable_controller_public_ips = false
  labels                        = var.labels
  network_self_link             = module.network1.network_self_link
  partition                     = flatten([module.v100_partition.partition, flatten([module.a2_partition.partition, flatten([module.debug_partition.partition])])])
  project_id                    = var.project_id
  region                        = var.region
  subnetwork_self_link          = module.network1.subnetwork_self_link
  zone                          = var.zone
}

module "slurm_login" {
  source                   = "./modules/embedded/community/modules/scheduler/schedmd-slurm-gcp-v5-login"
  controller_instance_id   = module.slurm_controller.controller_instance_id
  deployment_name          = var.deployment_name
  disable_login_public_ips = false
  labels                   = var.labels
  machine_type             = "n2-standard-4"
  network_self_link        = module.network1.network_self_link
  project_id               = var.project_id
  pubsub_topic             = module.slurm_controller.pubsub_topic
  region                   = var.region
  subnetwork_self_link     = module.network1.subnetwork_self_link
  zone                     = var.zone
}
