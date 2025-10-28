hookeai.material\_model\_finder.model.material\_discovery.MaterialModelFinder
=============================================================================

.. currentmodule:: hookeai.material_model_finder.model.material_discovery

.. autoclass:: MaterialModelFinder
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: List of Public Methods

   .. autosummary::
      :nosignatures:
   
      ~MaterialModelFinder.add_module
      ~MaterialModelFinder.apply
      ~MaterialModelFinder.bfloat16
      ~MaterialModelFinder.buffers
      ~MaterialModelFinder.build_element_local_samples
      ~MaterialModelFinder.build_elements_local_samples
      ~MaterialModelFinder.build_tensor_from_comps
      ~MaterialModelFinder.check_force_equilibrium_loss_type
      ~MaterialModelFinder.check_model_in_normalized
      ~MaterialModelFinder.check_model_out_normalized
      ~MaterialModelFinder.children
      ~MaterialModelFinder.compute_dirichlet_sets_reaction
      ~MaterialModelFinder.compute_dirichlet_sets_reaction_hist
      ~MaterialModelFinder.compute_element_internal_forces_hist
      ~MaterialModelFinder.cpu
      ~MaterialModelFinder.cuda
      ~MaterialModelFinder.double
      ~MaterialModelFinder.enforce_parameters_bounds
      ~MaterialModelFinder.enforce_parameters_constraints
      ~MaterialModelFinder.eval
      ~MaterialModelFinder.extra_repr
      ~MaterialModelFinder.features_out_extractor
      ~MaterialModelFinder.float
      ~MaterialModelFinder.force_equilibrium_loss
      ~MaterialModelFinder.force_equilibrium_loss_components_hist
      ~MaterialModelFinder.forward
      ~MaterialModelFinder.forward_sequential_element
      ~MaterialModelFinder.forward_sequential_time
      ~MaterialModelFinder.get_buffer
      ~MaterialModelFinder.get_detached_model_parameters
      ~MaterialModelFinder.get_device
      ~MaterialModelFinder.get_extra_state
      ~MaterialModelFinder.get_material_models
      ~MaterialModelFinder.get_model_parameters_bounds
      ~MaterialModelFinder.get_parameter
      ~MaterialModelFinder.get_submodule
      ~MaterialModelFinder.half
      ~MaterialModelFinder.ipu
      ~MaterialModelFinder.load_state_dict
      ~MaterialModelFinder.modules
      ~MaterialModelFinder.named_buffers
      ~MaterialModelFinder.named_children
      ~MaterialModelFinder.named_modules
      ~MaterialModelFinder.named_parameters
      ~MaterialModelFinder.parameters
      ~MaterialModelFinder.recurrent_material_state_update
      ~MaterialModelFinder.register_backward_hook
      ~MaterialModelFinder.register_buffer
      ~MaterialModelFinder.register_forward_hook
      ~MaterialModelFinder.register_forward_pre_hook
      ~MaterialModelFinder.register_full_backward_hook
      ~MaterialModelFinder.register_full_backward_pre_hook
      ~MaterialModelFinder.register_load_state_dict_post_hook
      ~MaterialModelFinder.register_module
      ~MaterialModelFinder.register_parameter
      ~MaterialModelFinder.register_state_dict_pre_hook
      ~MaterialModelFinder.requires_grad_
      ~MaterialModelFinder.set_device
      ~MaterialModelFinder.set_extra_state
      ~MaterialModelFinder.set_fitted_force_data_scalers
      ~MaterialModelFinder.set_material_models_fitted_data_scalers
      ~MaterialModelFinder.set_specimen_data
      ~MaterialModelFinder.share_memory
      ~MaterialModelFinder.state_dict
      ~MaterialModelFinder.store_dirichlet_sets_reaction_hist
      ~MaterialModelFinder.store_force_equilibrium_loss_components_hist
      ~MaterialModelFinder.store_tensor_comps
      ~MaterialModelFinder.to
      ~MaterialModelFinder.to_empty
      ~MaterialModelFinder.train
      ~MaterialModelFinder.type
      ~MaterialModelFinder.vassemble_internal_forces
      ~MaterialModelFinder.vbuild_internal_forces_mesh_hist
      ~MaterialModelFinder.vbuild_tensor_from_comps
      ~MaterialModelFinder.vcompute_element_internal_forces_hist
      ~MaterialModelFinder.vcompute_element_vol_grad_hist
      ~MaterialModelFinder.vcompute_elements_internal_forces_hist
      ~MaterialModelFinder.vcompute_local_dev_sym_gradient
      ~MaterialModelFinder.vcompute_local_gradient
      ~MaterialModelFinder.vcompute_local_internal_forces
      ~MaterialModelFinder.vcompute_local_internal_forces_hist
      ~MaterialModelFinder.vcompute_local_strain
      ~MaterialModelFinder.vcompute_local_strain_vbar
      ~MaterialModelFinder.vcompute_local_vol_grad_operator_hist
      ~MaterialModelFinder.vcompute_local_vol_sym_gradient
      ~MaterialModelFinder.vforce_equilibrium_hist_loss
      ~MaterialModelFinder.vforce_equilibrium_loss
      ~MaterialModelFinder.vforward_sequential_element
      ~MaterialModelFinder.vrecurrent_material_state_update
      ~MaterialModelFinder.vstore_tensor_comps
      ~MaterialModelFinder.xpu
      ~MaterialModelFinder.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~MaterialModelFinder.T_destination
      ~MaterialModelFinder.call_super_init
      ~MaterialModelFinder.dump_patches
      ~MaterialModelFinder.training
   
   

   .. rubric:: Methods