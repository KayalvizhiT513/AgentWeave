let eventSource = null;
let eventConversationId = null;

function connectLiveEvents(conversationId) {
  if (eventConversationId === conversationId) return;
  eventSource?.close();
  eventConversationId = conversationId;
  eventSource = new EventSource(`${api}/${conversationId}/events`);

  eventSource.onopen = () => setConnection(true);
  eventSource.onerror = () => setConnection(false);
  eventSource.onmessage = ({ data }) => {
    const event = JSON.parse(data);
    if (!current || event.conversation_id !== current.id) return;
    const { payload } = event;

    if (event.type === "conversation.agent_responded") {
      const exchange = payload.exchange;
      if (!current.exchanges.some(item => item.id === exchange.id)) current.exchanges.push(exchange);
      const index = current.agents.findIndex(agent => agent.id === payload.agent.id);
      if (index >= 0) current.agents[index] = payload.agent;
      current.current_round = Math.max(current.current_round, exchange.round_number);
    } else if (event.type === "conversation.evaluation_created") {
      if (!current.evaluations.some(item => item.id === payload.evaluation.id)) current.evaluations.push(payload.evaluation);
    } else if (event.type === "conversation.agent_replaced") {
      if (!current.replacements.some(item => item.id === payload.replacement.id)) current.replacements.push(payload.replacement);
      load(current.id);
    } else if (event.type === "conversation.round_completed") {
      current.current_round = payload.round;
      current.status = payload.status;
    } else if (event.type === "conversation.completed") {
      current.status = "completed";
      current.final_summary = payload.final_summary;
      eventSource.close();
      eventConversationId = null;
    }
    render();
  };
}

setInterval(() => {
  if (current && eventConversationId !== current.id) connectLiveEvents(current.id);
}, 150);
