import { useState, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000/api/v1";

// ─── Mock Data for Demo ──────────────────────────────────────
const MOCK_STATS = {
  total_conversations: 147,
  active_conversations: 23,
  waiting_response: 18,
  resolved: 98,
  escalated: 8,
  total_followups: 312,
  pending_followups: 45,
  completed_followups: 234,
  total_contacts: 89,
  rules: 6,
  channels: { chat: 34, email: 52, whatsapp: 71, voice: 12, sms: 0 },
  available_channels: ["chat", "email", "whatsapp", "voice"],
  total_state_bindings: 67,
  active_state_bindings: 31,
  state_distribution: {
    payment_collection: { pending: 4, reminded: 8, acknowledged: 5, promised: 6, confirmed: 12, overdue: 2, escalated: 1 },
    order_fulfillment: { placed: 3, confirmed: 5, shipped: 4, delivered: 7, resolved: 8, issue_reported: 1 },
    feedback_collection: { requested: 3, received: 5, processed: 8 },
  },
};

const MOCK_STATE_BINDINGS = [
  { id: "sb-001", process_type: "payment_collection", current_state: "promised", entity_type: "invoice", entity_id: "INV-2024-1234", contact_name: "Rajesh Kumar", is_terminal: false, transitions: 3, last_transition: "acknowledged → promised", updated_at: "2024-12-08T11:16:00" },
  { id: "sb-002", process_type: "order_fulfillment", current_state: "shipped", entity_type: "order", entity_id: "ORD-2024-5678", contact_name: "Priya Sharma", is_terminal: false, transitions: 3, last_transition: "processing → shipped", updated_at: "2024-12-12T14:30:00" },
  { id: "sb-003", process_type: "payment_collection", current_state: "confirmed", entity_type: "invoice", entity_id: "INV-2024-0987", contact_name: "Amit Patel", is_terminal: true, transitions: 4, last_transition: "promised → confirmed", updated_at: "2024-12-10T09:20:00" },
  { id: "sb-004", process_type: "payment_collection", current_state: "overdue", entity_type: "invoice", entity_id: "INV-2024-1111", contact_name: "Suresh Nair", is_terminal: false, transitions: 3, last_transition: "promised → overdue", updated_at: "2024-12-11T16:45:00" },
  { id: "sb-005", process_type: "feedback_collection", current_state: "received", entity_type: "order", entity_id: "ORD-2024-4321", contact_name: "Priya Sharma", is_terminal: false, transitions: 1, last_transition: "requested → received", updated_at: "2024-12-13T10:00:00" },
  { id: "sb-006", process_type: "order_fulfillment", current_state: "issue_reported", entity_type: "order", entity_id: "ORD-2024-7777", contact_name: "Vikram Singh", is_terminal: false, transitions: 4, last_transition: "delivered → issue_reported", updated_at: "2024-12-14T08:30:00" },
];

const MOCK_CONVERSATIONS = [
  {
    id: "conv-001",
    contact_id: "D001",
    status: "active",
    active_channel: "whatsapp",
    channels_used: ["whatsapp"],
    business_context: {
      type: "payment_reminder",
      invoice_number: "INV-2024-1234",
      amount: 250000,
      currency: "INR",
    },
    messages: [
      { direction: "outbound", content: "Hi Rajesh! Reminder about payment of INR 2,50,000 for INV-2024-1234. Due on Dec 1st.", channel: "whatsapp", timestamp: "2024-12-08T10:30:00" },
      { direction: "inbound", content: "Hi, we'll process it by end of this week. Had some cash flow issues.", channel: "whatsapp", timestamp: "2024-12-08T11:15:00" },
      { direction: "outbound", content: "Thank you Rajesh. We understand. Shall I note Friday Dec 13 as the expected payment date?", channel: "whatsapp", timestamp: "2024-12-08T11:16:00" },
    ],
    attempt_count: 2,
    created_at: "2024-12-05T09:00:00",
    updated_at: "2024-12-08T11:16:00",
    contact: { name: "Rajesh Kumar", role: "dealer", organization: "Kumar Motors" },
  },
  {
    id: "conv-002",
    contact_id: "D002",
    status: "waiting_response",
    active_channel: "email",
    channels_used: ["whatsapp", "email"],
    business_context: {
      type: "order_confirmation",
      order_id: "ORD-2024-5678",
      order_value: 500000,
    },
    messages: [
      { direction: "outbound", content: "Hi Priya! Your order ORD-2024-5678 has been confirmed.", channel: "whatsapp", timestamp: "2024-12-06T14:00:00" },
      { direction: "outbound", content: "Following up on order confirmation. Sent detailed breakdown via email.", channel: "email", timestamp: "2024-12-07T10:00:00" },
    ],
    attempt_count: 2,
    created_at: "2024-12-06T14:00:00",
    updated_at: "2024-12-07T10:00:00",
    contact: { name: "Priya Sharma", role: "dealer", organization: "Sharma Auto" },
  },
  {
    id: "conv-003",
    contact_id: "V001",
    status: "resolved",
    active_channel: "voice",
    channels_used: ["email", "voice"],
    business_context: {
      type: "delivery_tracking",
      po_number: "PO-2024-9012",
      shipment_id: "SHIP-3456",
    },
    messages: [
      { direction: "outbound", content: "Following up on PO-2024-9012 delivery status.", channel: "email", timestamp: "2024-12-04T09:00:00" },
      { direction: "outbound", content: "Called to discuss shipment SHIP-3456. Confirmed ETA Dec 15.", channel: "voice", timestamp: "2024-12-05T11:30:00" },
      { direction: "inbound", content: "Shipment on track. Will arrive Dec 15 morning.", channel: "voice", timestamp: "2024-12-05T11:35:00" },
    ],
    attempt_count: 2,
    created_at: "2024-12-04T09:00:00",
    updated_at: "2024-12-05T11:35:00",
    contact: { name: "AutoParts Global", role: "vendor", organization: "AutoParts Global Ltd" },
  },
  {
    id: "conv-004",
    contact_id: "D003",
    status: "escalated",
    active_channel: "email",
    channels_used: ["whatsapp", "email", "voice"],
    business_context: {
      type: "payment_reminder",
      invoice_number: "INV-2024-5555",
      amount: 780000,
      currency: "INR",
      days_overdue: 21,
    },
    messages: [
      { direction: "outbound", content: "Payment reminder for INV-2024-5555", channel: "whatsapp", timestamp: "2024-11-25T10:00:00" },
      { direction: "outbound", content: "Second reminder sent via email", channel: "email", timestamp: "2024-11-28T10:00:00" },
      { direction: "outbound", content: "Called regarding overdue payment", channel: "voice", timestamp: "2024-12-02T14:00:00" },
      { direction: "inbound", content: "We dispute the invoice amount. Need to speak with management.", channel: "voice", timestamp: "2024-12-02T14:10:00" },
    ],
    attempt_count: 4,
    created_at: "2024-11-25T10:00:00",
    updated_at: "2024-12-02T14:10:00",
    contact: { name: "Amit Patel", role: "dealer", organization: "Patel Automobiles" },
  },
];

const MOCK_FOLLOWUPS = [
  { id: "fu-001", rule_id: "payment_reminder", contact_id: "D001", status: "awaiting_reply", priority: "high", reason: "Payment overdue 7 days", attempt_count: 2, max_attempts: 3, current_channel: "whatsapp", scheduled_at: "2024-12-05T09:00:00" },
  { id: "fu-002", rule_id: "order_confirmation", contact_id: "D002", status: "awaiting_reply", priority: "medium", reason: "Order confirmation follow-up", attempt_count: 2, max_attempts: 3, current_channel: "email", scheduled_at: "2024-12-06T14:00:00" },
  { id: "fu-003", rule_id: "delivery_tracking", contact_id: "V001", status: "completed", priority: "medium", reason: "Delivery tracking update", attempt_count: 2, max_attempts: 3, current_channel: "voice", scheduled_at: "2024-12-04T09:00:00" },
  { id: "fu-004", rule_id: "payment_reminder", contact_id: "D003", status: "escalated", priority: "urgent", reason: "Payment overdue 21 days", attempt_count: 4, max_attempts: 3, current_channel: "email", scheduled_at: "2024-11-25T10:00:00" },
  { id: "fu-005", rule_id: "feedback_collection", contact_id: "D004", status: "scheduled", priority: "low", reason: "Post-delivery feedback", attempt_count: 0, max_attempts: 2, current_channel: null, scheduled_at: "2024-12-10T10:00:00" },
];

const MOCK_RULES = [
  { id: "payment_reminder", name: "Payment Reminder", enabled: true, trigger: { type: "schedule", cron: "0 10 * * MON-FRI" }, conditions: [{ field: "payment_status", operator: "eq", value: "overdue" }], actions: [{ type: "start_conversation", channel_priority: ["whatsapp", "email", "voice"], template: "payment_reminder" }] },
  { id: "order_confirmation", name: "Order Confirmation", enabled: true, trigger: { type: "event", event_name: "order_placed" }, conditions: [{ field: "order_value", operator: "gte", value: 10000 }], actions: [{ type: "start_conversation", channel_priority: ["whatsapp", "chat"], template: "order_confirmation" }] },
  { id: "delivery_tracking", name: "Delivery Update", enabled: true, trigger: { type: "event", event_name: "shipment_dispatched" }, conditions: [], actions: [{ type: "start_conversation", channel_priority: ["whatsapp", "email"], template: "delivery_update" }] },
  { id: "feedback_collection", name: "Post-Delivery Feedback", enabled: false, trigger: { type: "event", event_name: "delivery_completed" }, conditions: [], actions: [{ type: "start_conversation", channel_priority: ["whatsapp", "chat"], template: "feedback_request", delay_minutes: 1440 }] },
];

// ─── Channel Icons ───────────────────────────────────────────
const ChannelIcon = ({ channel, size = 16 }) => {
  const icons = {
    whatsapp: (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      </svg>
    ),
    email: (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
      </svg>
    ),
    chat: (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="m3 21 1.9-5.7a8.5 8.5 0 1 1 3.8 3.8z" />
      </svg>
    ),
    voice: (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z" />
      </svg>
    ),
  };
  return <span style={{ display: "inline-flex", alignItems: "center" }}>{icons[channel] || icons.chat}</span>;
};

// ─── Status Badge ────────────────────────────────────────────
const StatusBadge = ({ status }) => {
  const styles = {
    active: { bg: "#dcfce7", color: "#166534", border: "#86efac" },
    waiting_response: { bg: "#fef3c7", color: "#92400e", border: "#fcd34d" },
    resolved: { bg: "#e0e7ff", color: "#3730a3", border: "#a5b4fc" },
    escalated: { bg: "#fee2e2", color: "#991b1b", border: "#fca5a5" },
    pending: { bg: "#f3f4f6", color: "#374151", border: "#d1d5db" },
    scheduled: { bg: "#f3f4f6", color: "#374151", border: "#d1d5db" },
    awaiting_reply: { bg: "#fef3c7", color: "#92400e", border: "#fcd34d" },
    completed: { bg: "#dcfce7", color: "#166534", border: "#86efac" },
    failed: { bg: "#fee2e2", color: "#991b1b", border: "#fca5a5" },
    in_progress: { bg: "#dbeafe", color: "#1e40af", border: "#93c5fd" },
  };
  const s = styles[status] || styles.pending;
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 10px",
      borderRadius: "9999px",
      fontSize: "11px",
      fontWeight: 600,
      letterSpacing: "0.03em",
      textTransform: "uppercase",
      background: s.bg,
      color: s.color,
      border: `1px solid ${s.border}`,
    }}>
      {status.replace(/_/g, " ")}
    </span>
  );
};

const PriorityDot = ({ priority }) => {
  const colors = { urgent: "#ef4444", high: "#f59e0b", medium: "#3b82f6", low: "#9ca3af" };
  return <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: colors[priority] || colors.medium, marginRight: 6 }} />;
};

// ─── Stat Card ───────────────────────────────────────────────
const StatCard = ({ label, value, accent, sub }) => (
  <div style={{
    background: "#fff",
    borderRadius: 12,
    padding: "20px 24px",
    border: "1px solid #e5e7eb",
    minWidth: 0,
  }}>
    <div style={{ fontSize: 12, color: "#6b7280", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
    <div style={{ fontSize: 32, fontWeight: 700, color: accent || "#111827", lineHeight: 1 }}>{value}</div>
    {sub && <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 6 }}>{sub}</div>}
  </div>
);

// ─── Conversation Detail Panel ──────────────────────────────
const ConversationDetail = ({ conversation, onClose }) => {
  if (!conversation) return null;
  return (
    <div style={{
      position: "fixed", top: 0, right: 0, bottom: 0, width: 480,
      background: "#fff", boxShadow: "-4px 0 24px rgba(0,0,0,0.12)",
      zIndex: 50, display: "flex", flexDirection: "column",
      borderLeft: "1px solid #e5e7eb",
    }}>
      <div style={{ padding: "20px 24px", borderBottom: "1px solid #e5e7eb", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 16 }}>{conversation.contact?.name || "Unknown"}</div>
          <div style={{ fontSize: 12, color: "#6b7280" }}>{conversation.contact?.organization} · {conversation.contact?.role}</div>
        </div>
        <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 20, color: "#9ca3af", padding: 4 }}>✕</button>
      </div>

      <div style={{ padding: "16px 24px", borderBottom: "1px solid #f3f4f6", display: "flex", gap: 12, flexWrap: "wrap" }}>
        <StatusBadge status={conversation.status} />
        {conversation.channels_used?.map(ch => (
          <span key={ch} style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "2px 8px", borderRadius: 6, background: "#f3f4f6", fontSize: 11, fontWeight: 500 }}>
            <ChannelIcon channel={ch} size={12} /> {ch}
          </span>
        ))}
        <span style={{ fontSize: 11, color: "#9ca3af" }}>Attempts: {conversation.attempt_count}</span>
      </div>

      {conversation.business_context && (
        <div style={{ padding: "12px 24px", borderBottom: "1px solid #f3f4f6", background: "#f9fafb" }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Business Context</div>
          {Object.entries(conversation.business_context).filter(([k]) => k !== "context_key").map(([k, v]) => (
            <div key={k} style={{ fontSize: 12, marginBottom: 2 }}>
              <span style={{ color: "#9ca3af" }}>{k.replace(/_/g, " ")}:</span>{" "}
              <span style={{ color: "#374151", fontWeight: 500 }}>{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ flex: 1, overflowY: "auto", padding: "16px 24px" }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>Messages</div>
        {conversation.messages?.map((msg, i) => (
          <div key={i} style={{
            marginBottom: 12,
            display: "flex",
            flexDirection: "column",
            alignItems: msg.direction === "outbound" ? "flex-end" : "flex-start",
          }}>
            <div style={{
              maxWidth: "85%",
              padding: "10px 14px",
              borderRadius: 12,
              fontSize: 13,
              lineHeight: 1.5,
              background: msg.direction === "outbound" ? "#2563eb" : "#f3f4f6",
              color: msg.direction === "outbound" ? "#fff" : "#374151",
              borderBottomRightRadius: msg.direction === "outbound" ? 4 : 12,
              borderBottomLeftRadius: msg.direction === "inbound" ? 4 : 12,
            }}>
              {msg.content}
            </div>
            <div style={{ fontSize: 10, color: "#9ca3af", marginTop: 3, display: "flex", alignItems: "center", gap: 4 }}>
              <ChannelIcon channel={msg.channel} size={10} />
              {new Date(msg.timestamp).toLocaleString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── Main App ────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [stats, setStats] = useState(MOCK_STATS);
  const [conversations, setConversations] = useState(MOCK_CONVERSATIONS);
  const [followups, setFollowups] = useState(MOCK_FOLLOWUPS);
  const [rules, setRules] = useState(MOCK_RULES);
  const [stateBindings, setStateBindings] = useState(MOCK_STATE_BINDINGS);
  const [selectedConv, setSelectedConv] = useState(null);
  const [filterStatus, setFilterStatus] = useState("all");

  const filteredConvs = filterStatus === "all"
    ? conversations
    : conversations.filter(c => c.status === filterStatus);

  const filteredFollowups = filterStatus === "all"
    ? followups
    : followups.filter(f => f.status === filterStatus);

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "conversations", label: "Conversations" },
    { id: "followups", label: "Follow-ups" },
    { id: "states", label: "Business States" },
    { id: "rules", label: "Rules" },
    { id: "channels", label: "Channels" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#f8f9fb", fontFamily: "'DM Sans', 'SF Pro Display', -apple-system, system-ui, sans-serif" }}>
      {/* Top Bar */}
      <div style={{
        background: "#fff",
        borderBottom: "1px solid #e5e7eb",
        padding: "0 32px",
        display: "flex",
        alignItems: "center",
        height: 56,
        justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg, #2563eb, #7c3aed)",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: "#fff", fontWeight: 800, fontSize: 14,
          }}>C</div>
          <span style={{ fontWeight: 700, fontSize: 16, color: "#111827", letterSpacing: "-0.02em" }}>ConverseAgent</span>
          <span style={{ fontSize: 11, color: "#9ca3af", background: "#f3f4f6", padding: "2px 8px", borderRadius: 4, fontWeight: 500 }}>Dealer Management</span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {stats.available_channels.map(ch => (
            <span key={ch} style={{
              display: "inline-flex", alignItems: "center", gap: 4,
              padding: "4px 10px", borderRadius: 6,
              background: "#f0fdf4", color: "#166534",
              fontSize: 11, fontWeight: 500,
            }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e" }} />
              <ChannelIcon channel={ch} size={12} /> {ch}
            </span>
          ))}
        </div>
      </div>

      {/* Tab Nav */}
      <div style={{ background: "#fff", borderBottom: "1px solid #e5e7eb", padding: "0 32px", display: "flex", gap: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => { setActiveTab(t.id); setFilterStatus("all"); }} style={{
            padding: "12px 20px",
            background: "none", border: "none", cursor: "pointer",
            fontSize: 13, fontWeight: 500,
            color: activeTab === t.id ? "#2563eb" : "#6b7280",
            borderBottom: activeTab === t.id ? "2px solid #2563eb" : "2px solid transparent",
            transition: "all 0.15s",
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "24px 32px", maxWidth: 1400, margin: "0 auto" }}>

        {/* ── OVERVIEW ─────────────────────────────────── */}
        {activeTab === "overview" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, marginBottom: 32 }}>
              <StatCard label="Active Conversations" value={stats.active_conversations} accent="#2563eb" sub={`${stats.waiting_response} waiting reply`} />
              <StatCard label="Pending Follow-ups" value={stats.pending_followups} accent="#f59e0b" sub={`${stats.total_followups} total`} />
              <StatCard label="Resolved" value={stats.resolved} accent="#22c55e" sub={`${stats.completed_followups} follow-ups completed`} />
              <StatCard label="Escalated" value={stats.escalated} accent="#ef4444" />
              <StatCard label="Total Contacts" value={stats.total_contacts} />
              <StatCard label="Active Rules" value={stats.rules} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              {/* Channel Distribution */}
              <div style={{ background: "#fff", borderRadius: 12, padding: 24, border: "1px solid #e5e7eb" }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 16 }}>Channel Distribution</div>
                {Object.entries(stats.channels).filter(([_, v]) => v > 0).map(([ch, count]) => (
                  <div key={ch} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, width: 100, fontSize: 13, color: "#374151" }}>
                      <ChannelIcon channel={ch} size={14} /> {ch}
                    </div>
                    <div style={{ flex: 1, background: "#f3f4f6", borderRadius: 4, height: 8, overflow: "hidden" }}>
                      <div style={{
                        height: "100%",
                        borderRadius: 4,
                        width: `${(count / Math.max(...Object.values(stats.channels))) * 100}%`,
                        background: ch === "whatsapp" ? "#25D366" : ch === "email" ? "#2563eb" : ch === "voice" ? "#7c3aed" : "#6b7280",
                        transition: "width 0.5s ease",
                      }} />
                    </div>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#374151", width: 40, textAlign: "right" }}>{count}</span>
                  </div>
                ))}
              </div>

              {/* Recent Activity */}
              <div style={{ background: "#fff", borderRadius: 12, padding: 24, border: "1px solid #e5e7eb" }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 16 }}>Recent Conversations</div>
                {MOCK_CONVERSATIONS.slice(0, 4).map(c => (
                  <div key={c.id} onClick={() => { setSelectedConv(c); }} style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "10px 0", borderBottom: "1px solid #f3f4f6", cursor: "pointer",
                  }}>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "#374151" }}>{c.contact?.name}</div>
                      <div style={{ fontSize: 11, color: "#9ca3af" }}>{c.business_context?.type?.replace(/_/g, " ")}</div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <ChannelIcon channel={c.active_channel} size={14} />
                      <StatusBadge status={c.status} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── CONVERSATIONS ────────────────────────────── */}
        {activeTab === "conversations" && (
          <div>
            <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
              {["all", "active", "waiting_response", "resolved", "escalated"].map(s => (
                <button key={s} onClick={() => setFilterStatus(s)} style={{
                  padding: "6px 14px", borderRadius: 8, border: "1px solid #e5e7eb",
                  background: filterStatus === s ? "#2563eb" : "#fff",
                  color: filterStatus === s ? "#fff" : "#374151",
                  fontSize: 12, fontWeight: 500, cursor: "pointer", transition: "all 0.15s",
                }}>
                  {s === "all" ? "All" : s.replace(/_/g, " ")}
                </button>
              ))}
            </div>

            <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e5e7eb", overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Contact</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Context</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Channels</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Status</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Attempts</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Last Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredConvs.map(c => (
                    <tr key={c.id} onClick={() => setSelectedConv(c)} style={{ borderBottom: "1px solid #f3f4f6", cursor: "pointer", transition: "background 0.1s" }}
                      onMouseEnter={e => e.currentTarget.style.background = "#f9fafb"}
                      onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                      <td style={{ padding: "12px 16px" }}>
                        <div style={{ fontWeight: 600, color: "#374151" }}>{c.contact?.name}</div>
                        <div style={{ fontSize: 11, color: "#9ca3af" }}>{c.contact?.organization}</div>
                      </td>
                      <td style={{ padding: "12px 16px", color: "#6b7280" }}>
                        {c.business_context?.type?.replace(/_/g, " ")}
                        {c.business_context?.invoice_number && <span style={{ fontSize: 11, color: "#9ca3af", display: "block" }}>{c.business_context.invoice_number}</span>}
                        {c.business_context?.order_id && <span style={{ fontSize: 11, color: "#9ca3af", display: "block" }}>{c.business_context.order_id}</span>}
                      </td>
                      <td style={{ padding: "12px 16px" }}>
                        <div style={{ display: "flex", gap: 6 }}>
                          {c.channels_used?.map(ch => (
                            <span key={ch} style={{
                              padding: "2px 6px", borderRadius: 4, background: "#f3f4f6",
                              display: "inline-flex", alignItems: "center", gap: 3,
                              fontSize: 11,
                              fontWeight: ch === c.active_channel ? 700 : 400,
                              color: ch === c.active_channel ? "#2563eb" : "#9ca3af",
                            }}>
                              <ChannelIcon channel={ch} size={11} />
                            </span>
                          ))}
                        </div>
                      </td>
                      <td style={{ padding: "12px 16px" }}><StatusBadge status={c.status} /></td>
                      <td style={{ padding: "12px 16px", color: "#6b7280" }}>{c.attempt_count}</td>
                      <td style={{ padding: "12px 16px", color: "#9ca3af", fontSize: 12 }}>{new Date(c.updated_at).toLocaleDateString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── FOLLOW-UPS ──────────────────────────────── */}
        {activeTab === "followups" && (
          <div>
            <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
              {["all", "scheduled", "awaiting_reply", "completed", "escalated", "failed"].map(s => (
                <button key={s} onClick={() => setFilterStatus(s)} style={{
                  padding: "6px 14px", borderRadius: 8, border: "1px solid #e5e7eb",
                  background: filterStatus === s ? "#2563eb" : "#fff",
                  color: filterStatus === s ? "#fff" : "#374151",
                  fontSize: 12, fontWeight: 500, cursor: "pointer",
                }}>
                  {s === "all" ? "All" : s.replace(/_/g, " ")}
                </button>
              ))}
            </div>

            <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e5e7eb", overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Priority</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Rule</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Reason</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Channel</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Status</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>Attempts</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredFollowups.map(f => (
                    <tr key={f.id} style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "12px 16px" }}><PriorityDot priority={f.priority} />{f.priority}</td>
                      <td style={{ padding: "12px 16px", fontWeight: 600, color: "#374151" }}>{f.rule_id.replace(/_/g, " ")}</td>
                      <td style={{ padding: "12px 16px", color: "#6b7280" }}>{f.reason}</td>
                      <td style={{ padding: "12px 16px" }}>
                        {f.current_channel ? (
                          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                            <ChannelIcon channel={f.current_channel} size={14} /> {f.current_channel}
                          </span>
                        ) : <span style={{ color: "#d1d5db" }}>—</span>}
                      </td>
                      <td style={{ padding: "12px 16px" }}><StatusBadge status={f.status} /></td>
                      <td style={{ padding: "12px 16px", color: "#6b7280" }}>
                        {f.attempt_count}/{f.max_attempts}
                        {f.attempt_count >= f.max_attempts && <span style={{ color: "#ef4444", fontSize: 11, marginLeft: 4 }}>MAX</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── BUSINESS STATES ──────────────────────────── */}
        {activeTab === "states" && (
          <div>
            {/* State Distribution by Process */}
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, color: "#1f2937" }}>State Distribution by Process</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 16 }}>
                {Object.entries(stats.state_distribution || {}).map(([process, states]) => {
                  const total = Object.values(states).reduce((a, b) => a + b, 0);
                  const colors = { pending: "#f59e0b", reminded: "#3b82f6", acknowledged: "#8b5cf6", promised: "#6366f1", confirmed: "#10b981", overdue: "#ef4444", escalated: "#dc2626", closed: "#6b7280", placed: "#f59e0b", processing: "#3b82f6", shipped: "#6366f1", delivered: "#10b981", issue_reported: "#ef4444", resolved: "#10b981", cancelled: "#6b7280", requested: "#f59e0b", received: "#3b82f6", processed: "#10b981", partial_paid: "#8b5cf6" };
                  return (
                    <div key={process} style={{ background: "#fff", borderRadius: 12, padding: 20, border: "1px solid #e5e7eb" }}>
                      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4, textTransform: "capitalize" }}>
                        {process.replace(/_/g, " ")}
                      </div>
                      <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 12 }}>{total} total bindings</div>
                      {/* Stacked bar */}
                      <div style={{ display: "flex", height: 10, borderRadius: 5, overflow: "hidden", marginBottom: 12, background: "#f3f4f6" }}>
                        {Object.entries(states).map(([state, count]) => (
                          <div key={state} title={`${state}: ${count}`} style={{
                            width: `${(count / total) * 100}%`,
                            background: colors[state] || "#9ca3af",
                            transition: "width 0.3s",
                          }} />
                        ))}
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px" }}>
                        {Object.entries(states).map(([state, count]) => (
                          <div key={state} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11 }}>
                            <div style={{ width: 8, height: 8, borderRadius: 2, background: colors[state] || "#9ca3af" }} />
                            <span style={{ color: "#6b7280" }}>{state}: <b>{count}</b></span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Active State Bindings Table */}
            <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e5e7eb", overflow: "hidden" }}>
              <div style={{ padding: "16px 20px", borderBottom: "1px solid #f3f4f6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontWeight: 700, fontSize: 14 }}>Active State Bindings</span>
                <span style={{ fontSize: 12, color: "#9ca3af" }}>{stats.active_state_bindings || 0} active / {stats.total_state_bindings || 0} total</span>
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase" }}>Process</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase" }}>Current State</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase" }}>Entity</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase" }}>Contact</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase" }}>Transitions</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#6b7280", fontSize: 11, textTransform: "uppercase" }}>Last Change</th>
                  </tr>
                </thead>
                <tbody>
                  {stateBindings.map(sb => {
                    const stateColors = { pending: "#f59e0b", reminded: "#3b82f6", acknowledged: "#8b5cf6", promised: "#6366f1", confirmed: "#10b981", overdue: "#ef4444", escalated: "#dc2626", shipped: "#6366f1", delivered: "#10b981", issue_reported: "#ef4444", received: "#3b82f6" };
                    const bg = stateColors[sb.current_state] || "#6b7280";
                    return (
                      <tr key={sb.id} style={{ borderBottom: "1px solid #f3f4f6" }}>
                        <td style={{ padding: "12px 16px", fontWeight: 600, textTransform: "capitalize" }}>
                          {sb.process_type.replace(/_/g, " ")}
                        </td>
                        <td style={{ padding: "12px 16px" }}>
                          <span style={{ padding: "2px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: `${bg}18`, color: bg }}>
                            {sb.current_state}
                          </span>
                          {sb.is_terminal && <span style={{ marginLeft: 6, fontSize: 10, color: "#10b981" }}>✓ DONE</span>}
                        </td>
                        <td style={{ padding: "12px 16px", fontFamily: "monospace", fontSize: 12, color: "#6b7280" }}>
                          {sb.entity_type}:{sb.entity_id}
                        </td>
                        <td style={{ padding: "12px 16px" }}>{sb.contact_name}</td>
                        <td style={{ padding: "12px 16px", color: "#6b7280" }}>{sb.transitions}</td>
                        <td style={{ padding: "12px 16px", fontSize: 12, color: "#9ca3af" }}>
                          {sb.last_transition}
                          <div style={{ fontSize: 10, marginTop: 2 }}>{new Date(sb.updated_at).toLocaleDateString()}</div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── RULES ───────────────────────────────────── */}
        {activeTab === "rules" && (
          <div>
            <div style={{ display: "grid", gap: 16 }}>
              {rules.map(r => (
                <div key={r.id} style={{
                  background: "#fff", borderRadius: 12, padding: 24,
                  border: "1px solid #e5e7eb",
                  opacity: r.enabled ? 1 : 0.6,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 15, color: "#111827" }}>{r.name}</div>
                      <div style={{ fontSize: 12, color: "#9ca3af" }}>ID: {r.id}</div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{
                        padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600,
                        background: r.trigger.type === "schedule" ? "#eff6ff" : "#fef3c7",
                        color: r.trigger.type === "schedule" ? "#1d4ed8" : "#92400e",
                      }}>
                        {r.trigger.type === "schedule" ? `⏰ ${r.trigger.cron}` : `⚡ ${r.trigger.event_name}`}
                      </span>
                      <span style={{
                        padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600,
                        background: r.enabled ? "#dcfce7" : "#f3f4f6",
                        color: r.enabled ? "#166534" : "#9ca3af",
                      }}>
                        {r.enabled ? "Active" : "Disabled"}
                      </span>
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: 24 }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Conditions</div>
                      {r.conditions.length === 0 ? (
                        <div style={{ fontSize: 12, color: "#d1d5db" }}>No conditions (always matches)</div>
                      ) : r.conditions.map((c, i) => (
                        <div key={i} style={{
                          fontSize: 12, color: "#374151",
                          padding: "4px 8px", background: "#f9fafb", borderRadius: 4, marginBottom: 4,
                          fontFamily: "monospace",
                        }}>
                          {c.field} <span style={{ color: "#2563eb" }}>{c.operator}</span> {JSON.stringify(c.value)}
                        </div>
                      ))}
                    </div>

                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Actions</div>
                      {r.actions.map((a, i) => (
                        <div key={i} style={{ fontSize: 12, marginBottom: 4 }}>
                          <span style={{ color: "#374151", fontWeight: 500 }}>{a.type.replace(/_/g, " ")}</span>
                          <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                            {a.channel_priority?.map(ch => (
                              <span key={ch} style={{
                                display: "inline-flex", alignItems: "center", gap: 3,
                                padding: "2px 6px", borderRadius: 4, background: "#f3f4f6",
                                fontSize: 11,
                              }}>
                                <ChannelIcon channel={ch} size={10} /> {ch}
                              </span>
                            ))}
                          </div>
                          {a.template && <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 2 }}>Template: {a.template}</div>}
                          {a.delay_minutes > 0 && <div style={{ fontSize: 11, color: "#9ca3af" }}>Delay: {a.delay_minutes}m</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── CHANNELS ────────────────────────────────── */}
        {activeTab === "channels" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 16 }}>
              {[
                { id: "chat", name: "Live Chat", desc: "Real-time WebSocket chat. Supports queued messages for offline contacts.", color: "#6b7280", status: "active", details: "WebSocket endpoint: /ws/chat/{contact_id}" },
                { id: "email", name: "Email", desc: "SMTP outbound, webhook inbound. HTML templates with reply threading.", color: "#2563eb", status: "active", details: "Templates: payment_reminder, order_confirmation, delivery_update, feedback_request" },
                { id: "whatsapp", name: "WhatsApp Business", desc: "Cloud API integration. Template messages for business-initiated conversations. Interactive buttons.", color: "#25D366", status: "active", details: "Webhook: /webhooks/whatsapp • Supports text, template, and interactive messages" },
                { id: "voice", name: "Voice (Pipecat)", desc: "Pipecat-powered voice pipeline. STT → LLM → TTS with interruption handling and DTMF.", color: "#7c3aed", status: "active", details: "Pipeline: Deepgram STT → Claude → ElevenLabs TTS • Telephony: Twilio SIP" },
              ].map(ch => (
                <div key={ch.id} style={{
                  background: "#fff", borderRadius: 12, padding: 24,
                  border: "1px solid #e5e7eb",
                  borderLeft: `4px solid ${ch.color}`,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{
                        width: 36, height: 36, borderRadius: 8,
                        background: `${ch.color}15`, display: "flex",
                        alignItems: "center", justifyContent: "center", color: ch.color,
                      }}>
                        <ChannelIcon channel={ch.id} size={18} />
                      </div>
                      <div style={{ fontWeight: 700, fontSize: 15 }}>{ch.name}</div>
                    </div>
                    <span style={{
                      padding: "3px 8px", borderRadius: 6, fontSize: 10, fontWeight: 600,
                      background: "#dcfce7", color: "#166534", textTransform: "uppercase",
                    }}>{ch.status}</span>
                  </div>
                  <p style={{ fontSize: 13, color: "#6b7280", lineHeight: 1.5, marginBottom: 12 }}>{ch.desc}</p>
                  <div style={{ fontSize: 11, color: "#9ca3af", padding: "8px 10px", background: "#f9fafb", borderRadius: 6, fontFamily: "monospace" }}>{ch.details}</div>
                  <div style={{ marginTop: 12, fontSize: 13, fontWeight: 600, color: "#374151" }}>
                    {stats.channels[ch.id] || 0} conversations
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Conversation Detail Slide-over */}
      {selectedConv && (
        <>
          <div
            onClick={() => setSelectedConv(null)}
            style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.2)", zIndex: 40 }}
          />
          <ConversationDetail conversation={selectedConv} onClose={() => setSelectedConv(null)} />
        </>
      )}
    </div>
  );
}
