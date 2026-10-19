        "No results exceeded the configured example limits."
    )

# ==========================================
# MONITORING GRAPH
# ==========================================

st.subheader("Monitoring Trends")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    filtered["Date"],
    filtered["CFU"],
    marker="o",
    linewidth=2,
    label="CFU Count"
)

ax.plot(
    filtered["Date"],
    filtered["Limit"],
    linestyle="--",
    label="Example Limit"
)

ax.set_title(
    f"Environmental Monitoring Trend - {location}"
)
ax.set_xlabel("Sampling Date")
ax.set_ylabel("CFU Count")
ax.legend()
ax.grid(True, alpha=0.3)

fig.autofmt_xdate()

st.pyplot(fig)
plt.close(fig)

# ==========================================
# RESULTS TABLE
# ==========================================

st.subheader("Monitoring Results")

st.dataframe(
    filtered,
    width="stretch"
)

# ==========================================
# AUTOMATED QUALITY REPORT
# ==========================================

st.subheader("Automated Quality Report")

report = (
    f"Environmental Monitoring Summary - {location}\n\n"
    f"Total samples analyzed: {total}\n"
    f"Average CFU count: {average:.2f}\n"
    f"Results above limit: {above}\n\n"
)

if above:
    report += "Results requiring review:\n"

    for _, row in above_limit.iterrows():
        report += (
            f"- {row['Date'].strftime('%Y-%m-%d')}: "
            f"{row['CFU']} CFU "
            f"(example limit: {row['Limit']})\n"
        )

    report += (
        "\nOne or more results exceeded the "
        "configured example limits. "
        "Further review is recommended."
    )
else:
    report += (
        "All results were within the configured "
        "example limits. No above-limit results "
        "were identified."
    )

st.text(report)

st.download_button(
    "Download Quality Report",
    data=report,
    file_name="quality_report.txt",
    mime="text/plain"
)

# ==========================================
# RULE-BASED INVESTIGATION ASSISTANT
# ==========================================

st.subheader("Quality Investigation Assistant")

question = st.selectbox(
    "What would you like to know?",
    [
        "Select a question",
        "Which samples exceeded the limit?",
        "What is the average CFU count?",
        "What is the highest CFU count?",
        "What should be reviewed?"
    ]
)

if question == "Which samples exceeded the limit?":
    if above:
        st.warning(
            f"{above} result(s) exceeded the "
            "configured example limit."
        )
        st.dataframe(above_limit, width="stretch")
    else:
        st.success("No above-limit results identified.")

elif question == "What is the average CFU count?":
    st.info(
        f"The average CFU count for {location} "
        f"is {average:.2f}."
    )

elif question == "What is the highest CFU count?":
    st.info(
        f"The highest result was {highest['CFU']} CFU "
        f"on {highest['Date'].strftime('%Y-%m-%d')}."
    )

elif question == "What should be reviewed?":
    if above:
        st.warning(
            "Review flagged samples, sampling records, "
            "and relevant process conditions. "
            "These results alone do not establish a root cause."
        )
    else:
        st.info(
            "No results exceeded the configured example limits."
        )

# ==========================================
# AI-POWERED ASSISTANT
# ==========================================

st.divider()
st.subheader("🤖 AI Quality Insights Assistant")

st.write(
    "Ask a question about the selected location's "
    "environmental monitoring data."
)

st.caption(
    "AI responses may contain errors. Verify all numerical "
    "claims against the results table. Do not use this tool "
