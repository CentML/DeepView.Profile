'use babel';

import EventType from './EventType';

// This object is the "source of truth" for all event definitions.
// We generate event classes from the data here.
const events = new Map([
  // Application-wide events
  ['Skyline', [
    'Opened',
    'Connected',
    'Requested Analysis',
    'Received Analysis',
  ]],

  // Events specific to interactions
  ['Interaction', [
    'Clicked Run Time Entry',
    'Clicked Memory Entry',
  ]],

  // Application-wide errors. We use a separate category to
  // aggregate them separately from regular Skyline events.
  ['Error', [
    'Connection Error',
    'Connection Timeout',
    'Analysis Error',
    'Protocol Error',
  ]],
]);

// Create EventType instances for each defined event
// under the chosen categories
const exportedEvents = {};
for (const [category, events] of events) {
  const eventInstances = {};
  events.forEach((eventAction) => {
    const key = eventAction.toUpperCase().split(' ').join('_');
    eventInstances[key] = EventType.of(category, eventAction);
  });
  exportedEvents[category] = eventInstances;
}

// Events are accessible by category and action name in this exported object.
// The action name is capitalized and spaces have been replaced with underscores.
//
// For example, the event with category "Skyline" and action "Requested Analysis"
// will be available as:
//
//   import Events from './telemetry/events';
//   const event = Events.Skyline.REQUESTED_ANALYSIS;
//
// To record an event, this event type instance is passed to the TelemetryClient's
// record method.
export default exportedEvents;
