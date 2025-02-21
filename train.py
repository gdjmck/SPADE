"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
import torch.cuda
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

if __name__ == '__main__':
    # parse options
    opt = TrainOptions().parse()

    if opt.profile:
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()

    # print options to help debugging
    print(' '.join(sys.argv))

    # load the dataset
    dataloader = data.create_dataloader(opt)

    # create trainer for our model
    trainer = Pix2PixTrainer(opt)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)

    for epoch in iter_counter.training_epochs():
        torch.cuda.empty_cache()
        iter_counter.record_epoch_start(epoch)
        # if epoch > 1:
        #     dataloader.dataset.condition_history.update_mean_and_stdvar()
        if opt.profile:
            pr.enable()
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)
                if opt.isTrain and opt.tf_log:
                    trainer.log_histogram(step_index=epoch * len(dataloader) + i, model_type='G')
                trainer.log_loss(loss_dict=trainer.g_losses, step_index=epoch * len(dataloader) + i, phase='G')

            # train discriminator
            trainer.run_discriminator_one_step(data_i)
            if opt.isTrain and opt.tf_log:
                trainer.log_histogram(step_index=epoch * len(dataloader) + i, model_type='D')
            trainer.log_loss(loss_dict=trainer.d_losses, step_index=epoch * len(dataloader) + i, phase='D')

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                visuals = OrderedDict([('input_label', data_i['mask'] if 'mask' in data_i else data_i['label']),
                                       ('synthesized_image', trainer.get_latest_generated()),
                                       ('real_image', data_i['image'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
           epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

        if opt.profile:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
            with open('iter_profile_stats.txt', 'w') as f:
                f.write(s.getvalue())
            break

    trainer.summary_writer.close()
    print('Training was successfully finished.')
