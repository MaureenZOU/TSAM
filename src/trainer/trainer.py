import os
import time

import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from base import BaseTrainer
from evaluate import get_fid_score, get_i3d_activations, init_i3d_model, evaluate_video_error
from utils.readers import save_frames_to_dir
from model.loss import AdversarialLoss
# from xy_scripts import demo_image


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self, model, losses, metrics,
        optimizer_g, optimizer_d_s, optimizer_d_t, resume, config,
        data_loader, valid_data_loader=None, lr_scheduler=None,
        train_logger=None, learn_mask=True, test_data_loader=None,
        pretrained_path=None
    ):
        super().__init__(
            model, losses, metrics, optimizer_g,
            optimizer_d_s, optimizer_d_t, resume, config, train_logger,
            pretrained_path
        )
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = self.config['visualization']['log_step']
        self.loss_gan_s_w = config['gan_losses']['loss_gan_spatial_weight']
        self.loss_gan_t_w = config['gan_losses']['loss_gan_temporal_weight']
        self.adv_loss_fn = AdversarialLoss()
        self.evaluate_score = config['trainer'].get('evaluate_score', True)
        self.store_gated_values = False
        self.printlog = False
        self.use_flow = config['use_flow']

        self.valid_length = config['validation']['valid_length']
        self.valid_interval = config['validation']['valid_interval']

        if self.test_data_loader is not None:
            self.toPILImage = ToPILImage()
            self.evaluate_test_warp_error = config.get('evaluate_test_warp_error', False)
            self.test_output_root_dir = os.path.join(self.checkpoint_dir, 'test_outputs')
        init_i3d_model()

    def _store_gated_values(self, out_dir):
        from model.blocks import GatedConv, GatedDeconv

        def save_target(child, out_subdir):
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)
            if isinstance(child, GatedConv):
                target = child.gated_values[0]
            elif isinstance(child, GatedDeconv):
                target = child.conv.gated_values[0]
            else:
                raise ValueError('should be gated conv or gated deconv')
            target = target.transpose(0, 1)
            for t in range(target.shape[0]):
                for c in range(target.shape[1]):
                    out_file = os.path.join(out_subdir, f'time{t:03d}_channel{c:04d}.png')
                    self.toPILImage(target[t, c: c + 1]).save(out_file)

        for key, child in self.model.generator.coarse_net.upsample_module.named_children():
            out_subdir = os.path.join(out_dir, f'upsample_{key}')
            save_target(child, out_subdir)
        for key, child in self.model.generator.coarse_net.downsample_module.named_children():
            out_subdir = os.path.join(out_dir, f'downsample_{key}')
            save_target(child, out_subdir)

    def get_evaluate_index(self, T, vid_len=15, valid_inter=[3, 12]):
        start = []
        end = []
        iter_start = []
        iter_end = []
        out_start = []
        out_end = []
        end_length = (vid_len - valid_inter[0])
        start_length = valid_inter[1]

        if T <= vid_len:
            start.append(0)
            end.append(T)
            iter_start.append(0)
            iter_end.append(T)
            out_start.append(0)
            out_end.append(T)
        else:
            start.append(0)
            end.append(start_length)
            start.append(valid_inter[0])
            end.append(vid_len)

            iter_start.append(0)
            iter_end.append(vid_len)
            iter_start.append(T - vid_len)
            iter_end.append(T)

            out_start.append(0)
            out_end.append(start_length)
            out_start.append(T - end_length)
            out_end.append(T)

            inter_block = max(0, (T - start_length - end_length)//(valid_inter[1] - valid_inter[0]) + 1)
            for i in range(0, inter_block):
                iter_start.append(start_length + i*(valid_inter[1] - valid_inter[0]) - valid_inter[0])
                iter_end.append(start_length + i*(valid_inter[1] - valid_inter[0]) - valid_inter[0] + vid_len)

                out_start.append(start_length + i*(valid_inter[1] - valid_inter[0]))
                out_end.append(start_length + (i+1)*(valid_inter[1] - valid_inter[0]))

                start.append(valid_inter[0])
                end.append(valid_inter[1])

        return start, end, iter_start, iter_end, out_start, out_end


    def index_data(self, data, start, end):
        input_data = {}
        for key in data.keys():
            if len(data[key]) > 0:
                if 'flow' in key:
                    input_data[key] = data[key][:,start:end-1,:,:,:]
                else:
                    input_data[key] = data[key][:,start:end,:,:,:]
        return input_data


    def _evaluate_data_loader(self, epoch=None, output_root_dir=None, data_loader=None, name='test'):
        total_length = 0
        total_warp_error = 0 if self.evaluate_test_warp_error else None
        total_error = 0
        total_psnr = 0
        total_ssim = 0
        total_p_dist = 0

        if output_root_dir is None:
            output_root_dir = self.test_output_root_dir
        val_log_dir = os.path.join(self.checkpoint_dir, 'val.log')
        if epoch is not None:
            output_root_dir = os.path.join(output_root_dir, f"epoch_{epoch}")
        output_root_dir = os.path.join(output_root_dir, name)

        output_i3d_activations = []
        real_i3d_activations = []
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                n,t,c,h,w = data['input_tensors'].shape
                start, end, iter_start, iter_end, out_start, out_end = self.get_evaluate_index(t, vid_len=self.valid_length, valid_inter=self.valid_interval)

                inputs_ = torch.zeros((t,c,h,w))
                outputs_ = torch.zeros((t,c,h,w))
                targets_ = torch.zeros((t,c,h,w))
                masks_ = torch.zeros((t,1,h,w))
                
                for m,n,i,j,p,q in zip(start, end, iter_start, iter_end, out_start, out_end):
                    input_ = self.index_data(data, i,j)
                    _, _, data_input, model_output = self._process_data(input_, validation=True)
                    inputs, outputs, targets, masks = self._unpack_data(data_input, model_output)
                    if self.store_gated_values:
                        out_dir = os.path.join(output_root_dir, 'gated_values', f'input_{batch_idx:04}')
                        self._store_gated_values(out_dir)
                    outputs = outputs.clamp(0, 1)

                    if self.evaluate_score:
                        # get i3d activation
                        output_i3d_activations.append(get_i3d_activations(outputs).cpu().numpy())
                        real_i3d_activations.append(get_i3d_activations(targets).cpu().numpy())

                    assert len(outputs) == 1  # Batch size = 1 for testing
                    inputs = inputs[0].cpu()
                    outputs = outputs[0].cpu()
                    targets = targets[0].cpu()
                    masks = masks[0].cpu()

                    inputs_[p:q,:,:,:] = inputs[m:n,:,:,:]
                    outputs_[p:q,:,:,:] = outputs[m:n,:,:,:]
                    targets_[p:q,:,:,:] = targets[m:n,:,:,:]
                    masks_[p:q,:,:,:] = masks[m:n,:,:,:]
                    
                    # if epoch is not None and epoch == 0:
                    #     # Save inputs to output_dir
                    #     output_dir = os.path.join(output_root_dir, 'inputs', f"input_{batch_idx:04}")
                    #     self.logger.debug(f"Saving batch {batch_idx} input to {output_dir}")
                    #     save_frames_to_dir([self.toPILImage(t) for t in inputs.cpu()], output_dir)

                inputs = inputs_
                outputs = outputs_
                targets = targets_
                masks = masks_
                
                # if epoch is not None and epoch == 0:
                #     # Save inputs to output_dir
                #     output_dir = os.path.join(output_root_dir, 'inputs', f"input_{batch_idx:04}")
                #     self.logger.debug(f"Saving batch {batch_idx} input to {output_dir}")
                #     save_frames_to_dir([self.toPILImage(t) for t in inputs.cpu()], output_dir)

                if epoch is not None and epoch % 5 == 0:
                    # Save test results to output_dir
                    output_dir = os.path.join(output_root_dir, f"result_{batch_idx:04}")
                    self.logger.debug(f"Saving batch {batch_idx} to {output_dir}")
                    save_frames_to_dir([self.toPILImage(t) for t in outputs], output_dir)

                if self.evaluate_score:
                    # Evaluate scores
                    warp_error, error, psnr_value, ssim_value, p_dist, length = \
                        self._evaluate_test_video(outputs, targets, masks)
                    if self.evaluate_test_warp_error:
                        total_warp_error += warp_error
                    total_error += error
                    total_ssim += ssim_value
                    total_psnr += psnr_value
                    total_p_dist += p_dist
                    total_length += length

        # if 'test_object' in output_root_dir:
        # demo_image(output_root_dir)

        if self.evaluate_score:
            output_i3d_activations = np.concatenate(output_i3d_activations, axis=0)
            real_i3d_activations = np.concatenate(real_i3d_activations, axis=0)
            fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
        else:
            fid_score = 0
            total_p_dist = [0]
            total_length = 1
        total_p_dist = total_p_dist[0]

        if epoch is not None:
            self.writer.set_step(epoch, name)
            self._write_images(
                inputs, outputs, targets, masks,
                model_output=model_output, data_input=data_input
            )
            if self.evaluate_test_warp_error:
                self.writer.add_scalar('test_warp_error', total_warp_error / total_length)
            self.writer.add_scalar('test_mse', total_error / total_length)
            self.writer.add_scalar('test_ssim', total_ssim / total_length)
            self.writer.add_scalar('test_psnr', total_psnr / total_length)
            self.writer.add_scalar('test_p_dist', total_p_dist / total_length)
            self.writer.add_scalar('test_fid_score', fid_score)

        test_folder = output_root_dir.split('/')[-1]
        out_line = test_folder + ' epoch: ' + str(epoch) + ', test_mse:' + str(total_error / total_length)[0:6] + ' test_ssim:' + str(total_ssim / total_length)[0:6] + ' test_psnr:' + str(total_psnr / total_length)[0:6] + ' test_fid_score:' + str(fid_score)[0:6] + '\n'
        with open(val_log_dir, 'a') as fout:
            fout.write(out_line)

        return total_warp_error, total_error, total_ssim, total_psnr, total_p_dist, total_length, fid_score

    def _write_images(
            self, inputs, outputs, targets, masks, output_edges=None,
            target_edges=None, model_output=None, data_input=None
    ):
        self.writer.add_image('input', make_grid(inputs.cpu(), nrow=3, normalize=False))
        self.writer.add_image('loss_mask', make_grid(masks.cpu(), nrow=3, normalize=False))
        self.writer.add_image(
            'output', make_grid(outputs.clamp(0, 1).cpu(), nrow=3, normalize=False))
        self.writer.add_image('gt', make_grid(targets.cpu(), nrow=3, normalize=False))
        self.writer.add_image('diff', make_grid(targets.cpu() - outputs.cpu(), nrow=3, normalize=True))
        self.writer.add_image('IO_diff', make_grid(inputs.cpu() - outputs.cpu(), nrow=3, normalize=True))
        try:
            output_edges = self.losses['loss_edge'][0].current_output_edges
            target_edges = self.losses['loss_edge'][0].current_target_edges
            self.writer.add_image('output_edge', make_grid(output_edges[0].cpu(), nrow=3, normalize=True))
            self.writer.add_image('target_edge', make_grid(target_edges[0].cpu(), nrow=3, normalize=True))
        except Exception:
            pass
        try:
            guidances = data_input['guidances']
            self.writer.add_image('guidances', make_grid(guidances[0].cpu(), nrow=3, normalize=True))
        except Exception:
            pass

        if model_output is not None:
            if 'imcomplete_video' in model_output.keys():
                self.writer.add_image('imcomplete_video', make_grid(
                    model_output['imcomplete_video'][0].transpose(0, 1).cpu(), nrow=3, normalize=False))

    def _evaluate_test_video(self, output, gt_frames, masks):
        gt_images = [self.toPILImage(gt) for gt in gt_frames]
        result_images = [self.toPILImage(result) for result in output]
        mask_images = [self.toPILImage(mask / 255) for mask in masks]
        return evaluate_video_error(
            result_images, gt_images, mask_images,
            flownet_checkpoint_path=None,
            evaluate_warping_error=self.evaluate_test_warp_error,
            printlog=self.printlog
        )

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _get_grad_mean_magnitude(self, output, optimizer):
        """
        Get mean magitude (absolute value) of gradient of output w.r.t params in the optimizer.
        This function is used to get a simple understanding over the impact of a loss.

        :output: usually the loss you want to compute gradient w.r.t params
        :optimizer: the optimizer who contains the parameters you care

        Note:
            This function will reset the gradient stored in paramerter, so please
            use it before <your loss>.backward()

        Example:
            > grad_magnitude = self._get_grad_mean_magnitude(
                  loss_recon * self.loss_recon_w, self.optimizer_g))
            > print(grad_magnitude)
        """
        optimizer.zero_grad()
        output.backward(retain_graph=True)
        all_grad = []
        for group in optimizer.param_groups:
            for p in group['params']:
                all_grad.append(p.grad.view(-1))
        value = torch.cat(all_grad).abs().mean().item()
        optimizer.zero_grad()
        return value

    def _get_edge_guidances(self, tensors):
        from utils.edge import get_edge
        guidances = []
        for batch_idx in range(tensors.size(0)):
            batch_edges = []
            for frame_idx in range(tensors.size(1)):
                edge = get_edge(
                    tensors[batch_idx, frame_idx:frame_idx + 1]
                )
                batch_edges.append(edge)
            guidances.append(torch.cat(batch_edges, dim=0))
        guidances = torch.stack(guidances)
        return guidances

    def _process_data(self, data, validation=False):
        inputs = data["input_tensors"].to(self.device)
        masks = data["mask_tensors"].to(self.device)
        targets = data["gt_tensors"].to(self.device)

        if self.use_flow == True:
            forward_flow = data["forward_flow"].to(self.device)
            backward_flow = data["backward_flow"].to(self.device)
            forward_flowmask = data["forward_flowmask"].to(self.device)
            backward_flowmask = data["backward_flowmask"].to(self.device)            

        # guidances = self._get_edge_guidances(targets).to(self.device) if 'edge' in data['guidance'] else None
        # guidances = data["guidances"].to(self.device) if len(data["guidances"]) > 0 else None
        guidances = None
        data_input = {
            "inputs": inputs,
            "masks": masks,
            "targets": targets,
            "guidances": guidances
        }

        if validation == True:
            gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
            inputs = inputs.repeat((gpu_num,1,1,1,1))
            masks = masks.repeat((gpu_num,1,1,1,1))
            targets = targets.repeat((gpu_num,1,1,1,1))

            if self.use_flow == True:
                forward_flow = forward_flow.repeat((gpu_num,1,1,1,1))
                backward_flow = backward_flow.repeat((gpu_num,1,1,1,1))
                forward_flowmask = forward_flowmask.repeat((gpu_num,1,1,1,1))
                backward_flowmask = backward_flowmask.repeat((gpu_num,1,1,1,1))

        if self.use_flow == True:
            flows = [forward_flow, backward_flow, forward_flowmask, backward_flowmask]
        else:
            flows = []

        non_gan_loss, model_output = self.model(inputs, masks, guidances, flows, targets, validation=validation, model='G_loss')

        if validation == True:
            model_output['outputs'] = model_output['outputs'][0:1,:]

        gan_loss = 0
        if self.loss_gan_t_w > 0 and validation == False:
            gan_loss = self.model(inputs, masks, guidances, flows, targets, model_output['outputs'].detach(), model='D_t_loss')
        
        return non_gan_loss, gan_loss, data_input, model_output

    def _unpack_data(self, data_input, model_output):
        # inputs, outputs, targets, masks = self._unpack_data(data_input, model_output)
        return (
            data_input['inputs'],
            model_output['outputs'] if 'refined_outputs' not in model_output.keys()
            else model_output['refined_outputs'],
            data_input['targets'],
            data_input['masks']
        )

    def _get_non_gan_loss(self, non_gan_loss):
        # Compute and write all non-GAN losses to tensorboard by for loop
        loss_total = 0

        # DEBUG
        # losses = self.model.module.losses
        losses = self.model.losses

        for key, value in non_gan_loss.items():
            if 'gan' not in key:
                loss_total += torch.mean(value)*losses[key]
                self.writer.add_scalar(f'{key}', torch.mean(value).item())
            elif 'gan' in key:
                loss_total += self.loss_gan_t_w*torch.mean(value)
                self.writer.add_scalar(f'{key}', torch.mean(value).item())
        return loss_total

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        epoch_start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):
            batch_start_time = time.time()

            # Set writer
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            non_gan_loss, gan_loss, data_input, model_output = self._process_data(data)
            inputs, outputs, targets, masks = self._unpack_data(data_input, model_output)

            loss_total = self._get_non_gan_loss(non_gan_loss)

            self.optimizer_g.zero_grad()

            # Uncomment these lines to see the gradient
            # grad_recon = self._get_grad_mean_magnitude(loss_recon, self.optimizer_g)
            # grad_vgg = self._get_grad_mean_magnitude(loss_vgg, self.optimizer_g)
            # grad_gan_s = self._get_grad_mean_magnitude(loss_gan_s, self.optimizer_g)
            # grad_gan_t = self._get_grad_mean_magnitude(loss_gan_t, self.optimizer_g)
            # self.logger.info(f"Grad: recon {grad_recon} vgg {grad_vgg} gan_s {grad_gan_s} gan_t {grad_gan_t}")
            loss_total.backward()
            self.optimizer_g.step()

            # Train spatial and temporal discriminators

            for d in ['t']:
                weight = getattr(self, f'loss_gan_{d}_w')
                optimizer = getattr(self, f'optimizer_d_{d}')

                if weight > 0:
                    optimizer.zero_grad()
                    loss_d = (
                        torch.mean(gan_loss['loss_real'])
                        + torch.mean(gan_loss['loss_fake'])
                    ) / 2
                    loss_d.backward()
                    optimizer.step()

                    self.writer.add_scalar(f'loss_d_{d}', loss_d.item())
                    self.writer.add_scalar('loss_gan_t', torch.mean(non_gan_loss['gan_losses']).item())

            self.writer.add_scalar('loss_total', loss_total.item())

            with torch.no_grad():
                total_loss += loss_total.item()
                total_metrics += self._eval_metrics(outputs, targets)

            if self.verbosity >= 2 and \
                    (batch_idx % self.log_step == 0 and epoch < 30) or \
                    batch_idx == 0:
                self.logger.info(
                    f'Epoch: {epoch} [{batch_idx * self.data_loader.batch_size}/{self.data_loader.n_samples} '
                    f' ({100.0 * batch_idx / len(self.data_loader):.0f}%)] '
                    f'loss_total: {loss_total.item():.3f}, '
                    f'BT: {time.time() - batch_start_time:.2f}s'
                )

                self._write_images(inputs[0], outputs[0], targets[0], masks[0],
                                   model_output=model_output, data_input=data_input)
                        
        log = {
            'epoch_time': time.time() - epoch_start_time,
            'loss_total': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.test_data_loader is not None and epoch % self.config['trainer']['test_freq'] == 0:
            log = self.evaluate_test_set(epoch=epoch, log=log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def evaluate_test_set(self, output_root_dir=None, epoch=None, log=None):
        # Insert breakpoint when Nan
        self.model.eval()
        if isinstance(self.test_data_loader, list):
            test_data_loaders = self.test_data_loader
        else:
            test_data_loaders = [self.test_data_loader]
        try:
            for i, data_loader in enumerate(test_data_loaders):
                name = data_loader.name if data_loader.name is not None else f'test{i}'
                total_warp_error, total_error, total_ssim, total_psnr, total_p_dist, total_length, fid_score = \
                    self._evaluate_data_loader(data_loader=data_loader, name=name,
                                               output_root_dir=output_root_dir, epoch=epoch)

                if log is not None:
                    log[f'{name}_p_dist'] = total_p_dist / total_length
                    log[f'{name}_fid_score'] = fid_score
                if self.printlog:
                    self.logger.info(f'test set name: {name}')
                    if self.evaluate_test_warp_error:
                        self.logger.info(f'test_warp_error: {total_warp_error / total_length}')
                    self.logger.info(f'test_mse: {total_error / total_length}')
                    self.logger.info(f'test_ssim: {total_ssim / total_length}')
                    self.logger.info(f'test_psnr: {total_psnr / total_length}')
                    self.logger.info(f'test_p_dist: {total_p_dist / total_length}')
                    self.logger.info(f'test_fid_score: {fid_score}\n')
        except Exception as err:
            self.logger.error(err, exc_info=True)
            breakpoint()  # NOQA
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        self.logger.info(f"Doing {epoch} validation ..")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                if epoch == 1 and batch_idx > 5:
                    continue
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                non_gan_loss, _, data_input, model_output = self._process_data(data, validation=True)
                inputs, outputs, targets, masks = self._unpack_data(data_input, model_output)

                loss_total = self._get_non_gan_loss(non_gan_loss)

                self.writer.add_scalar('loss_total', loss_total.item())
                total_val_loss += loss_total.item()
                total_val_metrics += self._eval_metrics(outputs, targets)

                if batch_idx % self.log_step == 0:
                    self._write_images(
                        inputs[0], outputs[0], targets[0], masks[0],
                        model_output=model_output, data_input=data_input
                    )

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
        }
