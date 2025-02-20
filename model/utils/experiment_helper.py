import os
import yaml

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.summary import create_file_writer

from model.utils.plotter import get_imshow_image

class ExperimentHelper():
    """
    Experiment Helper class for conducting experiment.
    An object of this class manages:
        - initializing/restoring model checkpoints with optimizer states,
        - logging and writing loss metrics and images to Tensorboard.

    USAGE:
        1. Define a new or existing (for restoring) experiment name.
        2. Construct {optimizer, model}.
        3. Get the number of steps for a single epoech from dataloader.
        4. Construct a helper class object.
        5. In your training loop, call the methods described in the below.

    Methods:
    --------
        - save_checkpoint():
            Save current model and optimizer states to checkpoint.
        - update_on_epoch_end():
            Update current epoch index, and loss metrics.
        - update_tr_loss(value, tb=True)->average_loss:
            Update loss value to return average loss within this epoch.
        - update_val_loss(value, tb=True)->average_loss:
            Update loss value to return average loss within this epoch.
        - update_minitest_acc(accs_by_scope, scopes, key, tb=True):
        - write_image_tensorboard(key, image_mtx):
            key: (str) ex) 'tr_sim_mtx'
            image: (2d numpy array)
    """
    def __init__(
            self,
            checkpoint_name,
            optimizer,
            model_to_checkpoint,
            best_loss="val_loss",
            max_to_keep=5,
            cfg=dict()):
        """

        Parameters
        ----------
            checkpoint_name : (str)
                Checkpoint name.
            optimizer : <tf.keras.optimizer>
                Assign a pre-constructed optimizer.
            model_to_checkpoint : <tf.keras.Model>
                Model to train.
            best_loss : (str), optional
                The loss to determine the best model. The default is 
                "val_loss". Can be either "val_loss" or "tr_loss".
            max_to_keep : (int), optional
                Maximum number of checkpoints to keep. The default is 10.
            cfg : (dict), optional
                Config file, if available. The default is {}.

        """

        assert best_loss in ["val_loss", "tr_loss"], \
            "best_loss must be either 'val_loss' or 'tr_loss'"

        # Save the config file
        self.cfg = cfg

        # Experiment settings
        self._checkpoint_name = checkpoint_name
        self._cfg_use_tensorboard = cfg['TRAIN']['TENSORBOARD']

        # Initialize parameters
        self.epoch = 1
        self.best_tr_loss = (self.epoch, 1e10)
        self.best_val_loss = (self.epoch, 1e10)
        self.best_loss_track = best_loss

        # Choose the root directory for logs
        if cfg['MODEL']['LOG_ROOT_DIR']:
            _root_dir = cfg['MODEL']['LOG_ROOT_DIR']
        else:
            _root_dir = './logs/'

        # Set the default directories
        self._log_dir = os.path.join(_root_dir, 'fit', f'{checkpoint_name}/')
        self._checkpoint_save_dir = os.path.join(_root_dir, 'checkpoint', f'{checkpoint_name}/')
        self._best_checkpoint_save_dir = os.path.join(_root_dir, 'best_checkpoint', f'{checkpoint_name}/')

        # Tensorboard writers for train and validation losses
        self._tr_summary_writer = create_file_writer(os.path.join(self._log_dir, 'train'))
        self._val_summary_writer = create_file_writer(os.path.join(self._log_dir, 'val'))

        # Track the learning rate
        self._lr_summary_writer = create_file_writer(os.path.join(self._log_dir, 'lr'))

        # Tensorboard writers for mini test accuracies if mini test is enabled
        if cfg['TRAIN']['MINI_TEST_IN_TRAIN']:
            self._minitest_summary_writer_dict = dict()
            for key in ['f', 'L2(f)', 'g(f)']:
                self._minitest_summary_writer_dict[key] = create_file_writer(
                    os.path.join(self._log_dir, 'mini_test', key))
            self._minitest_acc = None

        # Tensorboard writer for images
        self._image_writer = create_file_writer(os.path.join(self._log_dir, 'images'))

        # Logging loss and acc metrics
        self._tr_loss = K.metrics.Mean(name='train_loss')
        self._tr_loss.reset_states()
        self._val_loss = K.metrics.Mean(name='val_loss')
        self._val_loss.reset_states()

        # Assign optimizer and model to checkpoint
        self.optimizer = optimizer # assign, not to create.
        self._model_to_checkpoint = model_to_checkpoint # assign, not to create.

        # General Setup checkpoint and checkpoint manager
        self._checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model_to_checkpoint
            )
        self.c_manager = tf.train.CheckpointManager(
            checkpoint=self._checkpoint,
            directory=self._checkpoint_save_dir,
            max_to_keep=max_to_keep,
            step_counter=self.optimizer.iterations,
            )

        # Best Loss Setup checkpoint and checkpoint manager
        self._best_checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model_to_checkpoint
            )
        self.c_manager_best = tf.train.CheckpointManager(
            checkpoint=self._best_checkpoint,
            directory=self._best_checkpoint_save_dir,
            max_to_keep=3, # Keep top 3 best models
            step_counter=self.optimizer.iterations,
            )

        self.load_checkpoint()
        # TODO: can this overwrite the loaded checkpoint?
        if self._cfg_use_tensorboard:
            self.write_lr()

    def update_on_epoch_end(self, save_checkpoint_now=True):
        """ Update current epoch index, and loss metrics. """

        if save_checkpoint_now:
            self.save_checkpoint()

        # Update the best training loss
        avg_tr_loss = self._tr_loss.result()
        if avg_tr_loss < self.best_tr_loss[1]:
            self.best_tr_loss = (self.epoch, avg_tr_loss)
            # Save the model if the best loss is updated
            if self.best_loss_track=="tr_loss":
                self.c_manager_best.save()

        # Update the best validation loss
        avg_val_loss = self._val_loss.result()
        if avg_val_loss < self.best_val_loss[1]:
            self.best_val_loss = (self.epoch, avg_val_loss)
            # Save the model if the best loss is updated
            if self.best_loss_track=="val_loss":
                self.c_manager_best.save()

        # Reset loss metrics
        self._tr_loss.reset_states()
        self._val_loss.reset_states()
        self.epoch += 1

        # Save learning rate to tensorboard
        if self._cfg_use_tensorboard:
            self.write_lr()

    def update_tr_loss(self, value, tb=None):
        """
        Parameters
        ----------
            value : (float)
                Loss value of the current train step.
            tb : (bool), optional
                Write to tensorboard if set True. The default is True.

        Returns
        -------
            avg_val_loss: (float) 
                Cumulative-average validation loss within current epoch after
                current iteration.
        """

        if tb or (tb==None and self._cfg_use_tensorboard):
            with self._tr_summary_writer.as_default():
                tf.summary.scalar('loss', value, step=self.optimizer.iterations)

        # Average the loss over the epoch
        avg_tr_loss = self._tr_loss(value)

        return avg_tr_loss

    def update_val_loss(self, value, tb=None):
        """
        Parameters
        ----------
            value : (float)
                Loss value of the current validation step.
            tb : (bool), optional
                Write to tensorboard if set True. The default is True.

        Returns
        -------
            avg_val_loss: (float) 
                Cumulative-average validation loss within current epoch after
                current iteration.
        """

        if tb or (tb==None and self._cfg_use_tensorboard):
            with self._val_summary_writer.as_default():
                tf.summary.scalar('loss', value, step=self.optimizer.iterations)

        # Cumulative Average the loss over the epoch
        avg_val_loss = self._val_loss(value)

        return avg_val_loss

    def update_minitest_acc(self, accs_by_scope, scopes, key, tb=None):
        """
        Parameters
        ----------
        accs_by_scope : list(float)
            Accuracies listed by scope. ex) [80.3, 50.4,...]
        scope : list(int)
            DESCRIPTION.
        key: (str)
            A key string usually representing a specific layer output.
            'g(f)' or 'f' or 'f_postL2'.
        tb : (bool), optional
            Write to tensorboard if set True. The default is True.

        """
        self._minitest_acc = accs_by_scope
        if tb or (tb==None and self._cfg_use_tensorboard):
            with self._minitest_summary_writer_dict[key].as_default():
                for acc, scope in list(zip(accs_by_scope[0], scopes)): # [0] is top1_acc
                    tf.summary.scalar(f'acc_{scope}s', acc, step=self.optimizer.iterations)
        else:
            pass

    def write_image_tensorboard(self, key, image_mtx):
        """
        Generate and write an image to tensorboard.

        Parameters
        ----------
        key: (str)
            A key string to specify an image source. ex) 'tr_sim_mtx'
        image_mtx : (2D NumPy array)
            Any 2D matrix representing an image.

        """
        if self._cfg_use_tensorboard:
            # Generate images
            img = get_imshow_image(image_mtx, f'{key}, Epoch={self.epoch}')
            img_post_softmax = get_imshow_image(tf.nn.softmax(
            image_mtx, axis=1).numpy(),
            title=f'{key + "_softmax"}, Epoch={self.epoch}')
            # Write image to tensorboard
            with self._image_writer.as_default():
                tf.summary.image(key, img, step=self.optimizer.iterations)
                tf.summary.image(key + '_softmax', img_post_softmax,
                                 step=self.optimizer.iterations)
        else:
            pass; # Not implemented yet

    def write_lr(self):

        if isinstance(self.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = self.optimizer.lr(self.optimizer.iterations)
        else:
            current_lr = self.optimizer.lr

        with self._lr_summary_writer.as_default():
            tf.summary.scalar('lr', 
                              current_lr,
                              step=self.optimizer.iterations)

    def load_checkpoint(self):
        """ Try loading a saved checkpoint. If no checkpoint, initialize from
            scratch.
        """

        if self.c_manager.latest_checkpoint:
            tf.print("-----------Restoring model from {}-----------".format(
                self.c_manager.latest_checkpoint))
            status = self._checkpoint.restore(self.c_manager.latest_checkpoint)
            status.expect_partial()
            self.epoch = int(self.c_manager.latest_checkpoint.split(sep='ckpt-')[-1]) + 1
        else:
            tf.print("-----------Initializing model from scratch-----------")
            # Write the config file to the checkpoint directory
            for _dir in [self._checkpoint_save_dir, self._best_checkpoint_save_dir]:
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                # Save the config file
                with open(_dir + 'config.yaml', 'w') as f:
                    yaml.dump(self.cfg, f)

    def save_checkpoint(self):
        """Save current model and optimizer states to checkpoint."""

        self.c_manager.save()