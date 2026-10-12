//! Prepass resolve passes.
//!
//! Bevy renders depth and motion vectors at full physical resolution, but
//! MetalFX wants them at the content size the frame was actually rendered at.
//! Both are resolved with a fullscreen triangle rather than
//! `copy_texture_to_texture`, for different reasons: the motion prepass texture
//! lacks `COPY_SRC`, and the depth target needs a format change on the way.
//!
//! Each pass keeps its pipeline and bind group in a `Mutex` on the node. This
//! module is a child of `node`, so it reaches those private fields directly
//! without widening their visibility.

use bevy::render::render_resource::{RenderPassDescriptor, TextureView};
use bevy::render::renderer::{RenderContext, RenderDevice};

use super::{MetalFxUpscaleNode, ResolvePipeline};

impl MetalFxUpscaleNode {
    /// Resolve full-res motion vectors into the content-sized RG16Float texture.
    pub(super) fn resolve_motion(
        &self,
        device: &RenderDevice,
        render_context: &mut RenderContext,
        src_texture: &bevy::render::render_resource::Texture,
        content_motion_view: &TextureView,
        content_w: u32,
        content_h: u32,
    ) {
        let motion_attachment_texture = src_texture;
        let mut mr = self.motion_resolve.lock().unwrap();
        if mr.is_none() {
            let wgpu_dev = device.wgpu_device();
            let shader = wgpu_dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("motion_resolve_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("motion_resolve.wgsl").into()),
            });
            let bgl = wgpu_dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motion_resolve_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            let pipeline_layout =
                wgpu_dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("motion_resolve_layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    // wgpu 29 replaced `push_constant_ranges` with a single
                    // `immediate_size`. Neither resolve pipeline uses push
                    // constants, so this is zero rather than a translation.
                    immediate_size: 0,
                });
            let pipeline = wgpu_dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("motion_resolve_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            *mr = Some(ResolvePipeline {
                pipeline,
                bind_group_layout: bgl,
            });
        }
        let mr_ref = mr.as_ref().unwrap();

        // Texture identity is stable across frames; a fresh view ID is not.
        let mut mr_bg = self.motion_resolve_bind_group.lock().unwrap();
        let need_new =
            !matches!(&*mr_bg, Some((src_id, _)) if *src_id == motion_attachment_texture.id());
        if need_new {
            let src_motion_view = motion_attachment_texture
                .create_view(&bevy::render::render_resource::TextureViewDescriptor::default());
            let wgpu_dev = device.wgpu_device();
            let src_view_wgpu: &wgpu::TextureView = &src_motion_view;
            let bg = wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("motion_resolve_bg"),
                layout: &mr_ref.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view_wgpu),
                }],
            });
            *mr_bg = Some((motion_attachment_texture.id(), bg));
        }
        let bind_group = &mr_bg.as_ref().unwrap().1;

        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("metalfx_motion_resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: content_motion_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(&mr_ref.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_viewport(0.0, 0.0, content_w as f32, content_h as f32, 0.0, 1.0);
        pass.draw(0..3, 0..1);
    }

    /// Resolve full-res depth into the content-sized Depth32Float texture.
    ///
    /// Kept in its own scope by the caller: the render pass guard must drop
    /// before `as_hal_mut` runs for the MetalFX encode, or wgpu's snatch lock
    /// panics.
    pub(super) fn resolve_depth(
        &self,
        device: &RenderDevice,
        render_context: &mut RenderContext,
        src_texture: &bevy::render::render_resource::Texture,
        content_depth_view: &TextureView,
        content_w: u32,
        content_h: u32,
    ) {
        let depth_attachment_texture = src_texture;
        // Lazy-init depth resolve render pipeline.
        let mut dr = self.depth_resolve.lock().unwrap();
        if dr.is_none() {
            let wgpu_dev = device.wgpu_device();
            let shader = wgpu_dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("depth_resolve_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("depth_resolve.wgsl").into()),
            });
            let bgl = wgpu_dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("depth_resolve_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            let pipeline_layout =
                wgpu_dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("depth_resolve_layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    // wgpu 29 replaced `push_constant_ranges` with a single
                    // `immediate_size`. Neither resolve pipeline uses push
                    // constants, so this is zero rather than a translation.
                    immediate_size: 0,
                });
            let pipeline = wgpu_dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth_resolve_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            *dr = Some(ResolvePipeline {
                pipeline,
                bind_group_layout: bgl,
            });
        }
        let dr_ref = dr.as_ref().unwrap();

        // The destination is a render attachment, not part of this binding.
        // Reuse the source view and binding until the prepass texture changes.
        let mut dr_bg = self.depth_resolve_bind_group.lock().unwrap();
        let need_new_bg = !matches!(&*dr_bg, Some((src_id, _))
            if *src_id == depth_attachment_texture.id());
        if need_new_bg {
            let src_depth_view = depth_attachment_texture
                .create_view(&bevy::render::render_resource::TextureViewDescriptor::default());
            let wgpu_dev = device.wgpu_device();
            // Extract the raw wgpu TextureView from Bevy's wrapped type.
            let src_view_wgpu: &wgpu::TextureView = &src_depth_view;
            let bg = wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("depth_resolve_bg"),
                layout: &dr_ref.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view_wgpu),
                }],
            });
            *dr_bg = Some((depth_attachment_texture.id(), bg));
        }
        let bind_group = &dr_bg.as_ref().unwrap().1;

        // Dispatch depth resolve render pass.
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("metalfx_depth_resolve"),
                color_attachments: &[],
                depth_stencil_attachment: Some(
                    bevy::render::render_resource::RenderPassDepthStencilAttachment {
                        view: content_depth_view,
                        depth_ops: Some(bevy::render::render_resource::Operations {
                            // Clear to 1.0 (near plane in Bevy's reversed-Z).
                            // Safe default: out-of-viewport fragments read as near-plane
                            // rather than far-plane (infinity), preventing edge ghosting.
                            load: bevy::render::render_resource::LoadOp::Clear(1.0),
                            store: bevy::render::render_resource::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    },
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(&dr_ref.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_viewport(0.0, 0.0, content_w as f32, content_h as f32, 0.0, 1.0);
        pass.draw(0..3, 0..1);
        // pass drops here → render encoder ends
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::SystemState;
    use bevy::prelude::World;
    use bevy::render::render_resource::{Texture, TextureDescriptor, TextureViewDescriptor};
    use bevy::render::renderer::PendingCommandBuffers;

    #[test]
    #[ignore = "requires a real Metal device; run explicitly on macOS"]
    fn resolve_bindings_reuse_source_texture_and_invalidate_on_replacement() {
        bevy::tasks::ComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = bevy::tasks::block_on(instance.request_adapter(&Default::default()))
            .expect("Metal adapter required");
        let (raw, queue) = bevy::tasks::block_on(adapter.request_device(&Default::default()))
            .expect("Metal device required");
        let device = RenderDevice::new(raw);
        let texture = |format| -> Texture {
            device.create_texture(&TextureDescriptor {
                label: Some("resolve_cache_regression"),
                size: wgpu::Extent3d {
                    width: 16,
                    height: 16,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        let mut world = World::new();
        world.insert_resource(device.clone());
        world.init_resource::<PendingCommandBuffers>();
        let mut system = SystemState::<RenderContext>::new(&mut world);
        let node = MetalFxUpscaleNode::default();
        for motion in [false, true] {
            let format = if motion {
                wgpu::TextureFormat::Rg16Float
            } else {
                wgpu::TextureFormat::Depth32Float
            };
            let source = texture(format);
            let destination = texture(format).create_view(&TextureViewDescriptor::default());
            let resolve = |source: &Texture,
                           world: &mut World,
                           system: &mut SystemState<RenderContext>| {
                {
                    let mut context = system.get_mut(world).expect("render resources");
                    if motion {
                        node.resolve_motion(&device, &mut context, source, &destination, 16, 16);
                    } else {
                        node.resolve_depth(&device, &mut context, source, &destination, 16, 16);
                    }
                }
                system.apply(world);
                queue.submit(world.resource_mut::<PendingCommandBuffers>().finish());
                if motion {
                    node.motion_resolve_bind_group
                        .lock()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .1
                        .clone()
                } else {
                    node.depth_resolve_bind_group
                        .lock()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .1
                        .clone()
                }
            };
            let first = resolve(&source, &mut world, &mut system);
            for _ in 0..8 {
                assert_eq!(
                    first,
                    resolve(&source, &mut world, &mut system),
                    "stable prepass must reuse its binding"
                );
            }
            let replacement = texture(format);
            assert_ne!(
                first,
                resolve(&replacement, &mut world, &mut system),
                "same-size recreated texture must invalidate the binding"
            );
        }
    }
}
