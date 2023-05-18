import collections
import logging
import os

from deepview_profile.exceptions import NoConnectionError

import deepview_profile.protocol_gen.innpv_pb2 as pm

logger = logging.getLogger(__name__)

RequestContext = collections.namedtuple(
    'RequestContext',
    ['address', 'state', 'sequence_number'],
)

def _validate_paths(project_root, entry_point):
    if not os.path.isabs(project_root):
        logger.error(
            "The project root that DeepView received is not an absolute path. "
            "This is an unexpected error. Please report a bug."
        )
        logger.error("Current project root: %s", project_root)
        return False

    if os.path.isabs(entry_point):
        logger.error(
            "The entry point must be specified as a relative path to the "
            "current directory. Please double check that the entry point you "
            "are providing is a relative path.",
        )
        logger.error("Current entry point path: %s", entry_point)
        return False

    full_path = os.path.join(project_root, entry_point)
    if not os.path.isfile(full_path):
        logger.error(
            "Either the specified entry point is not a file or its path was "
            "specified incorrectly. Please double check that it exists and "
            "that its path is correct.",
        )
        logger.error("Current absolute path to entry point: %s", full_path)
        return False

    return True


class MessageHandler:
    def __init__(
        self,
        connection_manager,
        message_sender,
        analysis_request_manager,
    ):
        self._connection_manager = connection_manager
        self._message_sender = message_sender
        self._analysis_request_manager = analysis_request_manager

    def _handle_initialize_request(self, message, context):
        if context.state.initialized:
            self._message_sender.send_protocol_error(
                pm.ProtocolError.ErrorCode.ALREADY_INITIALIZED_CONNECTION,
                context,
            )
            return
        if message.protocol_version != 5:
            # We only support version 5 of the protocol. We do not guarantee
            # backward compatibility for v0.x.x releases.
            # Version 1 - v0.1.x
            # Version 2 - v0.2.x
            # Version 3 - v0.3.x
            # Version 4 - v0.4.x
            self._message_sender.send_protocol_error(
                pm.ProtocolError.ErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
                context,
            )
            self._connection_manager.remove_connection(context.address)
            logger.error(
                'DeepView is out of date. Please update to the latest versions '
                'of the DeepView command line interface and plugin.'
            )
            return


        if not _validate_paths(message.project_root,  message.entry_point):
            # Change this to the error related to
            self._message_sender.send_protocol_error(
                pm.ProtocolError.ErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
                context,
            )
            self._connection_manager.remove_connection(context.address)
            logger.error(
                'Invalid project root or entry point.'
            )
            return
        logger.info("Connection addr:(%s:%d)", *context.address)
        logger.info("Project Root:   %s", message.project_root)
        logger.info("Entry Point:    %s", message.entry_point)
        self._connection_manager.get_connection(context.address)\
            .set_project_paths(message.project_root, message.entry_point)

        context.state.initialized = True
        self._message_sender.send_initialize_response(context)

    def _handle_analysis_request(self, message, context):
        if not context.state.initialized:
            self._message_sender.send_protocol_error(
                pm.ProtocolError.ErrorCode.UNINITIALIZED_CONNECTION,
                context,
            )
            return

        self._analysis_request_manager.submit_request(message, context)

    def handle_message(self, raw_data, address):
        try:
            message = pm.FromClient()
            message.ParseFromString(raw_data)
            logger.debug('Received message from (%s:%d).', *address)

            state = self._connection_manager.get_connection_state(address)
            if not state.is_request_current(message):
                logger.debug('Ignoring stale message from (%s:%d).', *address)
                return
            state.update_sequence(message)

            message_type = message.WhichOneof('payload')
            if message_type is None:
                logger.warn('Received empty message from (%s:%d).', *address)
                return

            context = RequestContext(
                state=state,
                address=address,
                sequence_number=message.sequence_number,
            )

            if message_type == 'initialize':
                self._handle_initialize_request(
                    getattr(message, message_type), context)
            elif message_type == 'analysis':
                self._handle_analysis_request(
                    getattr(message, message_type), context)
            else:
                # If the protobuf was compiled properly, this block should
                # never be reached.
                raise AssertionError(
                    'Invalid message type "{}".'.format(message_type))
        except NoConnectionError:
            logger.debug(
                'Dropping message from (%s:%d) because it is no longer '
                'connected.',
                *address,
            )
        except Exception:
            logger.exception(
                'Processing message from (%s:%d) resulted in an exception.',
                *address,
            )
