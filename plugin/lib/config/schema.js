'use babel';

// Configuration schemas need to follow the specifications in the
// Atom Config documentation. A default value is **required**
// because we use it to populate our initial configuration state.
//
// https://flight-manual.atom.io/api/v1.43.0/Config/

export default {
  enableTelemetry: {
    title: 'Enable Usage Statistics Reporting',
    description: 'Allow usage statistics to be sent to the Skyline team to help improve Skyline.',
    type: 'boolean',
    default: true,
  },

  disableProfileOnSave: {
    title: 'Disable Profile on Save',
    description: 'Prevent Skyline from automatically profiling your model each time you save your code. Instead, you will be able to trigger profiling manually by clicking a button in Skyline.',
    type: 'boolean',
    default: false,
  },
};
